"""기술 접근성 진단: AI 크롤러가 페이지에 닿고, 읽고, 인용할 수 있는지 확인한다."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal
from urllib.parse import quote, unquote, urljoin, urlparse

from bs4.element import PreformattedString

from ..models import CheckResult, SiteSnapshot, Status
from ..page import ParsedPage

BotPurpose = Literal["search", "training"]


@dataclass(frozen=True)
class AIBot:
    name: str  # robots.txt의 User-agent 토큰
    operator: str
    purpose: BotPurpose
    description: str


AI_BOTS: tuple[AIBot, ...] = (
    # 검색·실시간 답변용: 막으면 해당 서비스의 AI 답변/검색 결과에서 사라진다
    AIBot("OAI-SearchBot", "OpenAI", "search", "ChatGPT 검색 결과에 사이트를 보여 주기 위한 수집"),
    AIBot("ChatGPT-User", "OpenAI", "search", "ChatGPT 사용자가 요청할 때 페이지를 실시간으로 읽음"),
    AIBot("Claude-SearchBot", "Anthropic", "search", "Claude 검색 결과 품질을 위한 수집"),
    AIBot("Claude-User", "Anthropic", "search", "Claude 사용자가 요청할 때 페이지를 실시간으로 읽음"),
    AIBot("PerplexityBot", "Perplexity", "search", "Perplexity 검색 결과에 사이트를 보여 주기 위한 수집"),
    AIBot("Perplexity-User", "Perplexity", "search", "Perplexity 사용자가 요청할 때 페이지를 실시간으로 읽음"),
    AIBot("Googlebot", "Google", "search", "Google 검색 색인 (AI 개요·AI 모드 포함)"),
    AIBot("Bingbot", "Microsoft", "search", "Bing 검색 및 Microsoft Copilot 답변용 색인"),
    # 모델 학습용: 막는 것은 정당한 사업적 선택이며 GEO 실패가 아니다
    AIBot("GPTBot", "OpenAI", "training", "OpenAI 생성형 AI 모델 학습용 수집"),
    AIBot("ClaudeBot", "Anthropic", "training", "Anthropic 생성형 AI 모델 학습용 수집"),
    AIBot(
        "Google-Extended",
        "Google",
        "training",
        "Gemini 모델 학습·Gemini 앱 답변 근거 활용 여부를 정하는 토큰 (Google 검색·AI 개요에는 영향 없음)",
    ),
    AIBot(
        "Applebot-Extended",
        "Apple",
        "training",
        "Apple 생성형 AI 모델 학습 활용 여부를 정하는 토큰 (Apple 검색 노출에는 영향 없음)",
    ),
    AIBot("CCBot", "Common Crawl", "training", "Common Crawl 공개 웹 아카이브 (여러 AI 모델의 학습 데이터로 쓰임)"),
    AIBot("Meta-ExternalAgent", "Meta", "training", "Meta AI 모델 학습 등을 위한 수집"),
)

SEARCH_BOTS: tuple[AIBot, ...] = tuple(b for b in AI_BOTS if b.purpose == "search")
TRAINING_BOTS: tuple[AIBot, ...] = tuple(b for b in AI_BOTS if b.purpose == "training")

PURPOSE_LABELS: dict[str, str] = {"search": "검색·실시간 답변", "training": "모델 학습"}

# 검색 크롤러 점수 계산 시 상대 중요도 (나머지는 1)
_SEARCH_IMPORTANCE: dict[str, float] = {"googlebot": 3.0, "bingbot": 2.0, "oai-searchbot": 2.0}

MIN_TEXT_CHARS = 200
SMALL_MAX_SNIPPET = 200
_ROBOTS_MAX_CHARS = 500 * 1024  # Google과 같은 상한


# ---------------------------------------------------------------------------
# robots.txt 평가 (RFC 9309 + Google 구현 방식)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotsRule:
    allow: bool
    pattern: str  # 정규화(퍼센트 인코딩)된 경로 패턴
    line: str  # 사람이 읽을 원래 규칙, 예: "Disallow: /private"
    regex: re.Pattern


@dataclass(frozen=True)
class RobotsGroup:
    agents: tuple[str, ...]  # 소문자 제품 토큰, 전체 대상은 "*"
    rules: tuple[RobotsRule, ...]


@dataclass(frozen=True)
class RobotsDecision:
    allowed: bool
    group: str | None  # 적용된 그룹: 봇 이름, "*", 또는 None(적용 그룹 없음)
    rule: str | None  # 결정을 내린 규칙 줄, 없으면 None


_KEY_ALIASES = {
    "useragent": "user-agent",
    "user agent": "user-agent",
    "user_agent": "user-agent",
    "dissallow": "disallow",
    "dissalow": "disallow",
    "disalow": "disallow",
    "site-map": "sitemap",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")
_URL_SAFE = "".join(chr(c) for c in range(0x21, 0x7F))  # 공백 제외 출력 가능한 ASCII


def _product_token(value: str) -> str | None:
    value = (value or "").strip()
    if value.startswith("*"):
        return "*"
    m = _TOKEN_RE.match(value)
    return m.group(0).lower() if m else None


def _normalize_path(path: str) -> str:
    encoded = quote(path, safe=_URL_SAFE, errors="replace")
    return _PCT_RE.sub(lambda m: m.group(0).upper(), encoded)


def _compile_pattern(pattern: str) -> re.Pattern:
    anchored = pattern.endswith("$")
    if anchored:
        pattern = pattern[:-1]
    body = ".*".join(re.escape(part) for part in pattern.split("*"))
    return re.compile(body + (r"\Z" if anchored else ""), re.DOTALL)


@lru_cache(maxsize=64)
def _parse_robots_cached(robots_txt: str) -> tuple[RobotsGroup, ...]:
    text = robots_txt[:_ROBOTS_MAX_CHARS].lstrip("﻿")
    groups: list[tuple[list[str], list[RobotsRule]]] = []
    current: tuple[list[str], list[RobotsRule]] | None = None
    in_rules = False

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        key = _KEY_ALIASES.get(key, key)
        value = value.strip()

        if key == "user-agent":
            if current is None or in_rules:
                current = ([], [])
                groups.append(current)
                in_rules = False
            token = _product_token(value)
            if token:
                current[0].append(token)
        elif key in ("allow", "disallow"):
            if current is None:
                continue  # 첫 User-agent 앞의 규칙은 무시
            in_rules = True
            if not value:
                continue  # 빈 Disallow는 "모두 허용"
            pattern = value if value.startswith(("/", "*")) else "/" + value
            pattern = _normalize_path(pattern)
            label = "Allow" if key == "allow" else "Disallow"
            current[1].append(RobotsRule(key == "allow", pattern, f"{label}: {value}", _compile_pattern(pattern)))
        # sitemap, crawl-delay 등은 그룹을 끊지 않는다

    return tuple(RobotsGroup(tuple(agents), tuple(rules)) for agents, rules in groups)


def parse_robots(robots_txt: str | None) -> list[RobotsGroup]:
    """robots.txt를 User-agent 그룹 목록으로 파싱한다."""
    if not robots_txt:
        return []
    return list(_parse_robots_cached(robots_txt))


def robots_sitemaps(robots_txt: str | None) -> list[str]:
    """robots.txt의 Sitemap: 줄에 적힌 주소 목록."""
    found: list[str] = []
    for raw in (robots_txt or "")[:_ROBOTS_MAX_CHARS].splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if _KEY_ALIASES.get(key.strip().lower(), key.strip().lower()) == "sitemap" and value.strip():
            found.append(value.strip())
    return found


def _robots_path(url_or_path: str) -> str:
    value = (url_or_path or "").strip()
    if "://" in value:
        try:
            parsed = urlparse(value)
        except ValueError:
            return "/"
        path = parsed.path or "/"
        return path + ("?" + parsed.query if parsed.query else "")
    value = value.split("#", 1)[0]
    if not value:
        return "/"
    return value if value.startswith("/") else "/" + value


def evaluate_robots(robots_txt: str | None, user_agent: str, path: str) -> RobotsDecision:
    """user_agent(제품 토큰, 예: "GPTBot")가 path(경로 또는 전체 URL)를 수집해도 되는지 판단한다."""
    if not robots_txt or not robots_txt.strip():
        return RobotsDecision(True, None, None)
    groups = _parse_robots_cached(robots_txt)
    token = _product_token(user_agent) or (user_agent or "").strip().lower()

    matched = [g for g in groups if token in g.agents]
    group_name: str | None = (user_agent or "").strip()
    if not matched:
        matched = [g for g in groups if "*" in g.agents]
        group_name = "*"
    if not matched:
        return RobotsDecision(True, None, None)

    target = _normalize_path(_robots_path(path))
    best: RobotsRule | None = None
    for group in matched:  # 같은 봇에 해당하는 그룹이 여럿이면 규칙을 합친다
        for rule in group.rules:
            if not rule.regex.match(target):
                continue
            if (
                best is None
                or len(rule.pattern) > len(best.pattern)
                or (len(rule.pattern) == len(best.pattern) and rule.allow and not best.allow)
            ):
                best = rule
    return RobotsDecision(best is None or best.allow, group_name, best.line if best else None)


def is_allowed(robots_txt: str | None, user_agent: str, path: str) -> bool:
    """robots.txt가 없거나 비어 있으면 True."""
    return evaluate_robots(robots_txt, user_agent, path).allowed


def blocked_bots(
    robots_txt: str | None, path: str, purpose: BotPurpose | None = None
) -> list[tuple[AIBot, RobotsDecision]]:
    """AI_BOTS 중 path를 수집할 수 없는 봇과 그 판단 근거."""
    out: list[tuple[AIBot, RobotsDecision]] = []
    for bot in AI_BOTS:
        if purpose and bot.purpose != purpose:
            continue
        decision = evaluate_robots(robots_txt, bot.name, path)
        if not decision.allowed:
            out.append((bot, decision))
    return out


# ---------------------------------------------------------------------------
# 공통 도우미
# ---------------------------------------------------------------------------


def _clip(items: list[str], limit: int = 5, length: int = 200) -> list[str]:
    out: list[str] = []
    for item in items:
        item = re.sub(r"\s+", " ", str(item)).strip()
        if not item:
            continue
        out.append(item if len(item) <= length else item[: length - 1] + "…")
        if len(out) >= limit:
            break
    return out


def _result(
    name: str,
    title: str,
    status: Status,
    score: float,
    weight: float,
    detail: str,
    fix: str | None = None,
    evidence: list[str] | None = None,
) -> CheckResult:
    return CheckResult(
        id=f"technical.{name}",
        category="technical",
        title=title,
        status=status,
        score=round(max(0.0, min(1.0, score)), 3),
        weight=weight,
        detail=detail,
        fix=fix,
        evidence=_clip(evidence or []),
    )


def _safe(name: str, title: str, fn: Callable[[], list[CheckResult]]) -> list[CheckResult]:
    try:
        return fn()
    except Exception as e:  # 예상 못 한 HTML 하나로 전체 진단이 멈추지 않게 한다
        return [
            _result(
                name,
                title,
                "info",
                0.0,
                0.0,
                f"이 항목을 확인하는 중 문제가 생겨 건너뛰었습니다 ({type(e).__name__}).",
                fix="잠시 후 다시 진단해 보세요. 계속되면 페이지 HTML이 올바른지 확인하세요.",
            )
        ]


def _header_values(headers: dict[str, str] | None, name: str) -> list[str]:
    name = name.lower()
    return [str(v) for k, v in (headers or {}).items() if str(k).lower() == name and v]


def _host(url: str | None) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower()
    except ValueError:
        return ""


def _root_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return "https://도메인"
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return "https://도메인"


def _join(names: list[str]) -> str:
    return ", ".join(names)


# ---------------------------------------------------------------------------
# 1. HTTP 기본
# ---------------------------------------------------------------------------

_WAF_HINT = (
    "브라우저에서는 열리는데 여기서만 실패한다면 방화벽·CDN(예: Cloudflare)의 봇 차단 설정이 "
    "자동 접속을 막고 있을 수 있습니다. AI 크롤러도 같은 이유로 막힐 수 있으니 보안 설정의 봇 차단 항목을 확인하세요."
)


def _check_http_status(snapshot: SiteSnapshot) -> list[CheckResult]:
    title = "페이지 정상 응답 (HTTP 200)"
    code = snapshot.status_code
    evidence: list[str] = []
    if snapshot.url and snapshot.final_url and snapshot.url != snapshot.final_url:
        evidence.append(f"리다이렉트: {snapshot.url} → {snapshot.final_url}")

    if snapshot.fetch_error:
        return [
            _result(
                "http_status",
                title,
                "fail",
                0.0,
                10,
                f"페이지를 불러오지 못했습니다: {snapshot.fetch_error}",
                fix="주소가 맞는지, 사이트가 지금 열려 있는지 확인하세요. " + _WAF_HINT,
                evidence,
            )
        ]
    if code == 200:
        return [_result("http_status", title, "pass", 1.0, 10, "페이지가 정상(200)으로 응답합니다.", None, evidence)]

    evidence.append(f"응답 코드: {code}")
    if code in (401, 403):
        detail = f"서버가 접근을 거부했습니다({code}). 로그인이 필요하거나 봇 차단 설정이 자동 접속을 막고 있습니다."
        fix = "공개해야 하는 페이지라면 로그인 없이 열리게 하세요. " + _WAF_HINT
    elif code in (404, 410):
        detail = f"페이지를 찾을 수 없습니다({code}). 없는 페이지는 AI 답변에 인용될 수 없습니다."
        fix = "주소가 맞는지 확인하고, 옮긴 페이지라면 새 주소로 301 리다이렉트를 설정하세요."
    elif code == 429:
        detail = "요청이 너무 많다는 응답(429)을 받았습니다. 크롤러는 이 응답을 받으면 수집을 미루거나 줄입니다."
        fix = "서버나 CDN의 요청 제한(rate limit)이 정상 크롤러까지 막지 않도록 기준을 완화하세요."
    elif 500 <= code < 600:
        detail = f"서버 오류({code})가 발생했습니다. 오류가 계속되면 크롤러가 페이지를 색인에서 뺄 수 있습니다."
        fix = "서버 로그를 확인해 오류 원인을 고치세요. 호스팅 업체나 개발자에게 이 응답 코드를 전달하면 됩니다."
    elif 200 < code < 300 or 300 <= code < 400:
        return [
            _result(
                "http_status",
                title,
                "warn",
                0.5,
                10,
                f"응답 코드가 {code}입니다. 크롤러가 가장 안정적으로 처리하는 것은 200 응답입니다.",
                "본문을 담은 페이지가 200으로 응답하도록 서버 설정을 확인하세요. 리다이렉트라면 최종 주소가 200을 주는지 확인하세요.",
                evidence,
            )
        ]
    else:
        detail = f"정상적인 응답을 받지 못했습니다(코드 {code})."
        fix = "주소가 맞는지, 서버가 정상 작동하는지 확인하세요. " + _WAF_HINT
    return [_result("http_status", title, "fail", 0.0, 10, detail, fix, evidence)]


def _check_https(snapshot: SiteSnapshot) -> list[CheckResult]:
    title = "HTTPS 보안 연결"
    final = snapshot.final_url or snapshot.url
    try:
        scheme = urlparse(final).scheme.lower()
        start_scheme = urlparse(snapshot.url).scheme.lower()
    except ValueError:
        scheme = start_scheme = ""
    if scheme == "https":
        return [_result("https", title, "pass", 1.0, 5, "HTTPS로 안전하게 연결됩니다.", None, [final])]
    if start_scheme == "https" and scheme == "http":
        detail = "HTTPS 주소가 보안 연결이 없는 HTTP 주소로 리다이렉트됩니다."
    else:
        detail = "페이지가 HTTPS가 아닌 HTTP로 제공됩니다. 검색엔진과 AI 서비스는 보안 연결된 페이지를 더 신뢰합니다."
    return [
        _result(
            "https",
            title,
            "fail",
            0.0,
            5,
            detail,
            "호스팅·CDN에서 무료 SSL 인증서(예: Let's Encrypt)를 켜고, http:// 주소는 https://로 301 리다이렉트하세요.",
            [final],
        )
    ]


# ---------------------------------------------------------------------------
# 2. robots.txt — AI 크롤러 접근
# ---------------------------------------------------------------------------


def _decision_text(bot: AIBot, decision: RobotsDecision) -> str:
    group = "모든 봇(*)" if decision.group == "*" else bot.name
    return f"{bot.name} ({bot.operator}) — 'User-agent: {group}' 그룹의 '{decision.rule}' 규칙"


def _check_search_bots(snapshot: SiteSnapshot, path: str) -> list[CheckResult]:
    title = "AI 검색·답변 크롤러 접근 (robots.txt)"
    robots = snapshot.robots_txt
    if robots is None:
        return [
            _result(
                "ai_search_bots",
                title,
                "pass",
                1.0,
                10,
                "robots.txt를 찾지 못했거나 읽을 수 없어 모든 크롤러가 허용된 것으로 간주됩니다. "
                "AI 검색 노출을 막는 설정은 없지만, robots.txt를 두면 사이트맵 위치를 알리고 크롤러별 정책을 분명히 할 수 있습니다. "
                "(방화벽·CDN의 봇 차단은 이 진단으로 확인할 수 없습니다.)",
                None,
                ["robots.txt 없음"],
            )
        ]

    blocked = blocked_bots(robots, path, "search")
    if not blocked:
        return [
            _result(
                "ai_search_bots",
                title,
                "pass",
                1.0,
                10,
                f"robots.txt가 이 페이지({path})에 대해 AI 검색·답변 크롤러("
                f"{_join([b.name for b in SEARCH_BOTS])})를 모두 허용합니다. "
                "(방화벽·CDN의 봇 차단은 이 진단으로 확인할 수 없습니다.)",
                None,
                [f"검사한 경로: {path}"],
            )
        ]

    total = sum(_SEARCH_IMPORTANCE.get(b.name.lower(), 1.0) for b in SEARCH_BOTS)
    lost = sum(_SEARCH_IMPORTANCE.get(b.name.lower(), 1.0) for b, _ in blocked)
    names = [b.name for b, _ in blocked]

    detail = (
        f"robots.txt가 이 페이지({path})에 대해 AI 검색·답변 크롤러 {len(blocked)}개를 막고 있습니다: {_join(names)}. "
        "이 크롤러가 막히면 해당 서비스의 AI 답변과 검색 결과에 이 페이지가 인용되기 어렵습니다."
    )
    if any(b.name == "Googlebot" for b, _ in blocked):
        detail += " 특히 Googlebot을 막으면 Google 검색은 물론 AI 개요·AI 모드에서도 빠집니다."

    via_star = [b.name for b, d in blocked if d.group == "*"]
    via_own = [(b, d) for b, d in blocked if d.group != "*"]
    fix_parts: list[str] = []
    if via_star:
        snippet = "\n".join(f"User-agent: {n}" for n in via_star) + "\nAllow: /"
        fix_parts.append(
            "robots.txt에 아래처럼 검색·답변 크롤러 전용 그룹을 추가하세요:\n"
            f"{snippet}\n"
            "전용 그룹이 생기면 'User-agent: *' 그룹의 규칙은 이 봇들에게 적용되지 않으므로, "
            "계속 막아야 할 경로(예: /admin)가 있다면 이 그룹에도 Disallow 줄로 다시 적어 주세요."
        )
    for bot, decision in via_own:
        fix_parts.append(f"'User-agent: {bot.name}' 그룹에서 '{decision.rule}' 줄을 지우거나 범위를 좁히세요.")
    fix = "\n".join(fix_parts)

    return [
        _result(
            "ai_search_bots",
            title,
            "fail",
            1.0 - lost / total,
            10,
            detail,
            fix,
            [_decision_text(b, d) for b, d in blocked],
        )
    ]


def _check_training_bots(snapshot: SiteSnapshot, path: str) -> list[CheckResult]:
    title = "AI 모델 학습용 크롤러 정책 (참고)"
    blocked = blocked_bots(snapshot.robots_txt, path, "training")
    if not blocked:
        return [
            _result(
                "ai_training_bots",
                title,
                "info",
                0.0,
                0.0,
                f"모델 학습용 크롤러({_join([b.name for b in TRAINING_BOTS])})가 모두 허용되어 있습니다. "
                "학습 데이터 제공 여부는 사업적 선택이며 점수에는 반영하지 않습니다. "
                "허용하면 향후 AI 모델이 브랜드·제품 정보를 더 잘 알 수도 있지만 그 효과는 보장되지 않습니다.",
                "지금 상태를 유지해도 됩니다. 학습 제공을 원하지 않는다면 검색용 크롤러는 그대로 둔 채 "
                "robots.txt에 'User-agent: GPTBot' 다음 줄에 'Disallow: /'처럼 학습용 봇만 막는 그룹을 추가할 수 있습니다.",
                [],
            )
        ]

    names = [b.name for b, _ in blocked]
    detail = (
        f"모델 학습용 크롤러 {len(blocked)}개가 차단되어 있습니다: {_join(names)}. "
        "학습용 크롤러 차단은 콘텐츠를 AI 학습에 제공할지에 대한 정당한 사업적 선택이며 GEO 실패가 아니므로 점수에 반영하지 않습니다. "
        "검색·답변용 크롤러가 허용되어 있다면 AI 검색 답변에는 계속 인용될 수 있습니다. "
        "반대로 허용하면 향후 모델이 브랜드·제품 정보를 더 잘 알 수도 있지만 그 효과는 보장되지 않습니다."
    )
    if "Google-Extended" in names:
        detail += (
            " 참고로 Google-Extended 차단은 Google 검색과 AI 개요에는 영향을 주지 않습니다"
            "(Gemini 앱에서 답변 근거로 쓰이는 데에는 영향이 있을 수 있습니다)."
        )
    return [
        _result(
            "ai_training_bots",
            title,
            "info",
            0.0,
            0.0,
            detail,
            "현재 선택을 유지해도 됩니다. 학습 제공을 원한다면 robots.txt에서 해당 봇의 Disallow 줄을 지우세요.",
            [_decision_text(b, d) for b, d in blocked],
        )
    ]


# ---------------------------------------------------------------------------
# 3. 색인·스니펫 지시어 (meta robots, X-Robots-Tag, data-nosnippet)
# ---------------------------------------------------------------------------

_COLON_DIRECTIVES = {"max-snippet", "max-image-preview", "max-video-preview", "unavailable_after"}
_RELEVANT_SCOPES = {"*", "robots", "googlebot", "bingbot", "msnbot"} | {b.name.lower() for b in AI_BOTS}
_SCOPED_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$")
_MAX_SNIPPET_RE = re.compile(r"^max-snippet\s*:\s*(-?\d+)")


@dataclass
class _Directives:
    by_scope: dict[str, set[str]]
    max_snippet: dict[str, int]
    evidence: list[str]

    def scopes_with(self, *names: str) -> list[str]:
        return sorted(s for s, d in self.by_scope.items() if s in _RELEVANT_SCOPES and d & set(names))


def _add_directives(out: _Directives, value: str, default_scope: str, allow_scope: bool) -> None:
    scope = default_scope
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        m = _SCOPED_RE.match(part)
        if allow_scope and m and m.group(1).lower() not in _COLON_DIRECTIVES:
            scope = m.group(1).lower()
            part = m.group(2).strip()
            if not part:
                continue
        directive = part.lower()
        key = directive.split(":", 1)[0].strip()
        bucket = out.by_scope.setdefault(scope, set())
        if key == "none":
            bucket.update({"noindex", "nofollow"})
        bucket.add(key)
        sm = _MAX_SNIPPET_RE.match(directive)
        if sm:
            n = int(sm.group(1))
            prev = out.max_snippet.get(scope)
            if prev is None or (n >= 0 and (prev < 0 or n < prev)):
                out.max_snippet[scope] = n  # 여러 값이면 가장 엄격한 값


def _collect_directives(snapshot: SiteSnapshot, page: ParsedPage) -> _Directives:
    out = _Directives({}, {}, [])
    for tag in page.soup.find_all("meta"):
        name = str(tag.get("name") or "").strip().lower()
        content = tag.get("content")
        if not name or not content:
            continue
        scope = "*" if name == "robots" else name
        if scope not in _RELEVANT_SCOPES:
            continue
        _add_directives(out, str(content), scope, allow_scope=False)
        out.evidence.append(f'<meta name="{tag.get("name")}" content="{content}">')
    for value in _header_values(snapshot.headers, "x-robots-tag"):
        _add_directives(out, value, "*", allow_scope=True)
        out.evidence.append(f"X-Robots-Tag: {value}")
    return out


def _scope_label(scopes: list[str]) -> str:
    if "*" in scopes:
        return "모든 검색엔진·크롤러"
    return _join(scopes)


def _check_indexing(snapshot: SiteSnapshot, page: ParsedPage) -> list[CheckResult]:
    directives = _collect_directives(snapshot, page)
    results: list[CheckResult] = []

    title = "검색 색인 허용 (noindex 없음)"
    noindex = directives.scopes_with("noindex")
    if noindex:
        results.append(
            _result(
                "indexing",
                title,
                "fail",
                0.0,
                10,
                f"이 페이지에 noindex 지시어가 있어 {_scope_label(noindex)}에 대해 색인이 금지되어 있습니다. "
                "색인되지 않은 페이지는 Google AI 개요, Copilot 같은 AI 검색 답변에 인용될 수 없습니다.",
                "공개해야 하는 페이지라면 <meta name=\"robots\" content=\"noindex\"> 태그나 X-Robots-Tag: noindex 응답 헤더를 지우세요. "
                "워드프레스 등 CMS라면 '검색엔진이 사이트를 색인하지 않도록' 같은 설정이 켜져 있는지 확인하세요.",
                directives.evidence,
            )
        )
    else:
        results.append(
            _result(
                "indexing",
                title,
                "pass",
                1.0,
                10,
                "noindex 지시어가 없어 검색엔진이 이 페이지를 색인할 수 있습니다.",
                None,
                directives.evidence,
            )
        )

    title = "AI 답변 인용 허용 (nosnippet·max-snippet)"
    nosnippet = directives.scopes_with("nosnippet")
    relevant_limits = {s: n for s, n in directives.max_snippet.items() if s in _RELEVANT_SCOPES and n >= 0}
    zero = sorted(s for s, n in relevant_limits.items() if n == 0)
    small = sorted(s for s, n in relevant_limits.items() if 0 < n < SMALL_MAX_SNIPPET)
    noarchive = directives.scopes_with("noarchive", "nocache")
    noarchive = [s for s in noarchive if s in ("*", "bingbot", "msnbot")]
    remove_fix = (
        "페이지 전체가 아니라 인용되면 안 되는 일부 영역만 숨기고 싶다면, 해당 요소에만 data-nosnippet 속성을 쓰세요."
    )

    if nosnippet or zero:
        scopes = sorted(set(nosnippet) | set(zero))
        results.append(
            _result(
                "snippet",
                title,
                "fail",
                0.0,
                8,
                f"{_scope_label(scopes)}에 대해 nosnippet 또는 max-snippet:0이 지정되어 있습니다. "
                "Google은 이 지시어를 AI 개요·AI 모드에도 적용하므로, 페이지 내용이 AI 답변에 인용되거나 요약되지 않습니다.",
                "robots 메타 태그나 X-Robots-Tag 헤더에서 nosnippet, max-snippet:0을 지우세요. " + remove_fix,
                directives.evidence,
            )
        )
    elif small:
        limit = min(relevant_limits[s] for s in small)
        results.append(
            _result(
                "snippet",
                title,
                "warn",
                0.5,
                8,
                f"max-snippet:{limit}로 인용 가능한 글자 수가 짧게 제한되어 있습니다. "
                "AI 답변이 페이지 내용을 충분히 인용·요약하기 어려울 수 있습니다.",
                "제한이 꼭 필요하지 않다면 max-snippet 값을 지우거나 max-snippet:-1(제한 없음)로 바꾸세요.",
                directives.evidence,
            )
        )
    elif noarchive:
        results.append(
            _result(
                "snippet",
                title,
                "warn",
                0.7,
                8,
                "noarchive 또는 nocache 지시어가 있습니다. Google은 noarchive를 더 이상 쓰지 않지만, "
                "Bing은 noarchive 페이지를 Copilot 답변에 쓰지 않고 nocache 페이지는 주소·제목·짧은 요약만 쓴다고 안내한 바 있습니다.",
                "Copilot 답변 노출을 원한다면 robots 메타 태그나 X-Robots-Tag에서 noarchive, nocache를 지우세요.",
                directives.evidence,
            )
        )
    else:
        results.append(
            _result(
                "snippet",
                title,
                "pass",
                1.0,
                8,
                "nosnippet·max-snippet 제한이 없어 AI 답변이 페이지 내용을 인용할 수 있습니다.",
                None,
                directives.evidence,
            )
        )

    hidden = page.soup.find_all(attrs={"data-nosnippet": True})
    if hidden:
        samples = []
        for el in hidden:
            text = el.get_text(" ", strip=True)
            samples.append(f"<{el.name} data-nosnippet> {text[:80]}" if text else f"<{el.name} data-nosnippet> (빈 요소)")
        results.append(
            _result(
                "data_nosnippet",
                "data-nosnippet 지정 영역 (참고)",
                "info",
                0.0,
                0.0,
                f"data-nosnippet으로 지정된 영역이 {len(hidden)}곳 있습니다. 이 영역의 글은 검색 스니펫과 Google AI 개요에 인용되지 않습니다. "
                "개인정보·내부 안내처럼 의도적으로 숨긴 것이라면 문제없습니다.",
                "서비스 소개, 가격, 자주 묻는 질문처럼 AI가 인용해도 좋은 핵심 내용에는 data-nosnippet을 쓰지 않았는지 확인하세요.",
                samples,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 4. 서버 렌더링 텍스트
# ---------------------------------------------------------------------------

_SPA_ROOT_IDS = ("root", "app", "__next", "__nuxt", "___gatsby", "svelte", "q-app")
_INVISIBLE_TAGS = {"script", "style", "noscript", "template", "svg"}
_JS_TYPES = {"", "module", "text/javascript", "application/javascript", "text/ecmascript", "application/ecmascript"}
_JS_REQUIRED_RE = re.compile(r"enable javascript|javascript (?:is )?(?:required|to run)|자바스크립트|javascript를", re.I)


def _is_js(tag) -> bool:
    kind = str(tag.get("type") or "").strip().lower()
    return kind in _JS_TYPES or "javascript" in kind


def _has_visible_text(el) -> bool:
    for s in el.find_all(string=True):
        if isinstance(s, PreformattedString) or not s.strip():
            continue
        if any(getattr(p, "name", None) in _INVISIBLE_TAGS for p in s.parents):
            continue
        return True
    return False


def _check_rendered_text(snapshot: SiteSnapshot, page: ParsedPage) -> list[CheckResult]:
    title = "자바스크립트 없이 읽히는 본문"
    soup = page.soup
    chars = page.char_count
    html_len = len(page.raw_html or snapshot.html or "")

    js_scripts = [s for s in soup.find_all("script") if _is_js(s)]
    empty_roots: list[str] = []
    for root_id in _SPA_ROOT_IDS:
        el = soup.find(id=root_id)
        if el is not None and not _has_visible_text(el):
            empty_roots.append(f'<{el.name} id="{root_id}">')
    for el in soup.find_all("app-root"):
        if not _has_visible_text(el):
            empty_roots.append("<app-root>")
    js_notice = [
        n.get_text(" ", strip=True) for n in soup.find_all("noscript") if _JS_REQUIRED_RE.search(n.get_text(" "))
    ]

    evidence = [f"자바스크립트 실행 전 보이는 텍스트: {chars:,}자", f"<script> {len(js_scripts)}개, HTML {html_len:,}자"]
    evidence += [f"비어 있는 앱 루트: {r}" for r in empty_roots]
    evidence += [f"<noscript> {t}" for t in js_notice]

    ssr_fix = (
        "서버 사이드 렌더링(SSR)이나 정적 생성(SSG/프리렌더링)을 켜서 처음 받는 HTML에 본문이 들어 있게 하세요. "
        "예: Next.js·Nuxt는 SSR/정적 생성 기능, React·Vue 단독 앱은 프리렌더링 도구를 사용합니다. "
        "웹빌더를 쓴다면 'SEO' 또는 '검색엔진 최적화' 설정을 확인하세요. "
        "브라우저에서 자바스크립트를 끄고 페이지를 열어 보면 AI 크롤러가 보는 화면을 확인할 수 있습니다."
    )

    if chars < MIN_TEXT_CHARS:
        spa_signals = bool(empty_roots or js_notice or len(js_scripts) >= 3 or html_len >= 10_000)
        if spa_signals:
            return [
                _result(
                    "server_rendered_text",
                    title,
                    "fail",
                    0.0,
                    8,
                    f"자바스크립트를 실행하기 전의 HTML에는 보이는 텍스트가 {chars}자뿐입니다. "
                    "화면 내용을 자바스크립트로 그리는 구조로 보이는데, 많은 AI 크롤러는 자바스크립트를 실행하지 않아 본문을 읽지 못할 수 있습니다.",
                    ssr_fix,
                    evidence,
                )
            ]
        return [
            _result(
                "server_rendered_text",
                title,
                "warn",
                0.4,
                8,
                f"페이지에서 읽을 수 있는 텍스트가 {chars}자로 매우 적습니다. AI가 인용할 설명이 거의 없습니다.",
                "서비스·제품을 설명하는 본문 텍스트를 HTML에 직접 추가하세요. 이미지 속 글자는 AI가 읽기 어렵습니다.",
                evidence,
            )
        ]
    if empty_roots:
        return [
            _result(
                "server_rendered_text",
                title,
                "warn",
                0.6,
                8,
                f"텍스트는 {chars:,}자 있지만 앱 루트 요소({_join(empty_roots)})가 비어 있어, "
                "주요 본문이 자바스크립트로 나중에 그려질 가능성이 있습니다. AI 크롤러는 그 부분을 읽지 못할 수 있습니다.",
                ssr_fix,
                evidence,
            )
        ]
    return [
        _result(
            "server_rendered_text",
            title,
            "pass",
            1.0,
            8,
            f"자바스크립트를 실행하지 않아도 HTML에서 텍스트 {chars:,}자를 읽을 수 있습니다.",
            None,
            evidence,
        )
    ]


# ---------------------------------------------------------------------------
# 5~6. llms.txt, 사이트맵
# ---------------------------------------------------------------------------

_LLMS_HONESTY = (
    "llms.txt는 AI에게 사이트 요약과 주요 페이지를 알려 주자는 새로운 제안(llmstxt.org)이며, "
    "주요 AI 검색 서비스가 이를 사용한다고 공식 확인한 바는 아직 없습니다."
)


def _check_llms_txt(snapshot: SiteSnapshot) -> list[CheckResult]:
    title = "llms.txt (선택 사항)"
    root = _root_url(snapshot.final_url or snapshot.url)
    content = (snapshot.llms_txt or "").strip()
    if content:
        first = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
        if first.startswith("# "):
            return [
                _result("llms_txt", title, "pass", 1.0, 2, "llms.txt가 있습니다. " + _LLMS_HONESTY, None, [first])
            ]
        return [
            _result(
                "llms_txt",
                title,
                "warn",
                0.7,
                2,
                "llms.txt가 있지만 첫 줄이 제안된 형식('# 사이트 이름' 제목)과 다릅니다. " + _LLMS_HONESTY,
                "첫 줄을 '# 사이트 이름'으로 시작하고, 이어서 '> 한 줄 소개'와 주요 페이지 링크 목록을 Markdown으로 적으세요.",
                [first],
            )
        ]
    return [
        _result(
            "llms_txt",
            title,
            "warn",
            0.0,
            2,
            "llms.txt가 없습니다. " + _LLMS_HONESTY + " 효과는 불확실하지만 만드는 비용이 낮아 추가해 둘 만합니다.",
            f"사이트 이름, 한두 문장 소개, 주요 페이지 링크 목록을 Markdown으로 정리해 {root}/llms.txt 에 올리세요.",
            [],
        )
    ]


def _check_sitemap(snapshot: SiteSnapshot) -> list[CheckResult]:
    title = "사이트맵"
    root = _root_url(snapshot.final_url or snapshot.url)
    declared = robots_sitemaps(snapshot.robots_txt)
    if declared or snapshot.sitemap_found:
        evidence = [f"Sitemap: {s}" for s in declared] or ["sitemap.xml 확인됨"]
        return [
            _result(
                "sitemap",
                title,
                "pass",
                1.0,
                3,
                "사이트맵이 있어 검색엔진과 AI 검색 크롤러가 페이지 목록을 쉽게 찾을 수 있습니다.",
                None,
                evidence,
            )
        ]
    return [
        _result(
            "sitemap",
            title,
            "warn",
            0.0,
            3,
            "사이트맵을 찾지 못했습니다(robots.txt의 Sitemap 줄과 /sitemap.xml 모두 없음). "
            "사이트맵은 검색엔진과 AI 검색 크롤러가 새 페이지를 빨리 찾는 데 도움이 됩니다.",
            f"CMS·웹빌더의 사이트맵 기능을 켜거나 sitemap.xml을 만들어 {root}/sitemap.xml 에 올리고, "
            f"robots.txt에 'Sitemap: {root}/sitemap.xml' 줄을 추가하세요. "
            "Google Search Console, 네이버 서치어드바이저, Bing 웹마스터 도구에도 제출하면 좋습니다.",
            [],
        )
    ]


# ---------------------------------------------------------------------------
# 7. html lang, canonical
# ---------------------------------------------------------------------------

_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})*$")


def _korean_dominant(text: str) -> bool:
    hangul = len(re.findall(r"[가-힣]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    return hangul >= 50 and hangul / (hangul + latin) > 0.5


def _check_lang(page: ParsedPage) -> list[CheckResult]:
    title = "페이지 언어 표시 (html lang)"
    lang = (page.lang or "").strip()
    korean = _korean_dominant(page.text)
    suggest = "ko" if korean else "ko 또는 en처럼 실제 본문 언어 코드"

    if not lang:
        return [
            _result(
                "html_lang",
                title,
                "warn",
                0.0,
                3,
                "<html> 태그에 lang 속성이 없습니다. 언어가 표시되어 있으면 검색엔진과 AI가 어떤 언어의 질문에 이 페이지를 보여 줄지 판단하기 쉽습니다.",
                f'<html lang="{suggest}">처럼 지정하세요.' if korean else f"<html lang=\"...\">에 {suggest}를 넣으세요.",
                [],
            )
        ]
    if "_" in lang or not _LANG_RE.match(lang):
        fixed = lang.replace("_", "-")
        return [
            _result(
                "html_lang",
                title,
                "warn",
                0.6,
                3,
                f'lang="{lang}"는 표준 언어 코드 형식이 아닙니다. ko, ko-KR, en처럼 하이픈(-)을 쓰는 형식이어야 합니다.',
                f'<html lang="{fixed if _LANG_RE.match(fixed) else "ko"}">처럼 고치세요.',
                [f'<html lang="{lang}">'],
            )
        ]
    if korean and lang.split("-")[0].lower() != "ko":
        return [
            _result(
                "html_lang",
                title,
                "warn",
                0.5,
                3,
                f'본문은 주로 한국어인데 lang="{lang}"로 지정되어 있습니다. 언어 정보가 틀리면 한국어 질문에 대한 답변에서 불리할 수 있습니다.',
                '<html lang="ko">로 고치세요. 다국어 사이트라면 언어별 페이지마다 맞는 코드를 쓰세요.',
                [f'<html lang="{lang}">'],
            )
        ]
    return [_result("html_lang", title, "pass", 1.0, 3, f'lang="{lang}"로 페이지 언어가 지정되어 있습니다.', None, [f'<html lang="{lang}">'])]


_LINK_CANONICAL_RE = re.compile(r"<([^>]+)>\s*;[^,]*?rel\s*=\s*\"?canonical\"?", re.I)


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def _norm_path(url: str) -> str:
    try:
        path = unquote(urlparse(url).path or "/")
    except ValueError:
        return "/"
    return path.rstrip("/") or "/"


def _check_canonical(snapshot: SiteSnapshot, page: ParsedPage) -> list[CheckResult]:
    title = "대표 주소 (canonical)"
    final = snapshot.final_url or snapshot.url
    try:
        clean = urlparse(final)._replace(query="", fragment="").geturl()
    except ValueError:
        clean = final

    tags = page.soup.find_all(
        "link", rel=lambda v: bool(v) and "canonical" in [r.lower() for r in (v if isinstance(v, list) else [v])]
    )
    hrefs: list[str] = []
    for tag in tags:
        href = str(tag.get("href") or "").strip()
        if href:
            hrefs.append(urljoin(page.base_url or final, href))
    source = "link 태그"
    if not hrefs:
        for value in _header_values(snapshot.headers, "link"):
            hrefs += [m.group(1).strip() for m in _LINK_CANONICAL_RE.finditer(value)]
        source = "Link 헤더"
    unique = list(dict.fromkeys(hrefs))
    evidence = [f'<link rel="canonical" href="{h}">' if source == "link 태그" else f"Link: <{h}>; rel=canonical" for h in unique]

    if not unique:
        return [
            _result(
                "canonical",
                title,
                "warn",
                0.3,
                4,
                "대표 주소(canonical)가 지정되어 있지 않습니다. 같은 내용이 여러 주소(파라미터, www 유무 등)로 열리면 "
                "검색엔진과 AI가 어느 주소를 대표로 인용할지 헷갈릴 수 있습니다.",
                f'<head> 안에 <link rel="canonical" href="{clean}"> 를 추가하세요.',
                [],
            )
        ]
    if len(unique) > 1:
        return [
            _result(
                "canonical",
                title,
                "warn",
                0.3,
                4,
                f"서로 다른 canonical이 {len(unique)}개 지정되어 있습니다. 이렇게 충돌하면 검색엔진이 모두 무시할 수 있습니다.",
                "canonical 태그를 하나만 남기세요. 테마와 SEO 플러그인이 각각 넣고 있지 않은지 확인하세요.",
                evidence,
            )
        ]

    canon = unique[0]
    canon_host, final_host = _host(canon), _host(final)
    if canon_host and final_host and _strip_www(canon_host) != _strip_www(final_host):
        return [
            _result(
                "canonical",
                title,
                "warn",
                0.2,
                4,
                f"canonical이 다른 도메인({canon_host})을 가리킵니다. 이 페이지 대신 그 도메인의 페이지가 색인·인용될 수 있습니다.",
                f'의도한 것이 아니라면 canonical을 이 페이지 주소로 고치세요: <link rel="canonical" href="{clean}">',
                evidence,
            )
        ]
    if _norm_path(canon) != _norm_path(final):
        return [
            _result(
                "canonical",
                title,
                "warn",
                0.6,
                4,
                f"canonical이 다른 페이지({canon})를 대표 주소로 지정합니다. 그러면 검색엔진은 이 페이지 대신 그 페이지를 색인합니다. "
                "모든 페이지가 홈 주소를 canonical로 쓰는 CMS 설정 오류가 흔합니다.",
                f'이 페이지가 독립된 내용이라면 canonical을 자기 주소로 고치세요: <link rel="canonical" href="{clean}">',
                evidence,
            )
        ]
    try:
        canon_scheme, final_scheme = urlparse(canon).scheme.lower(), urlparse(final).scheme.lower()
    except ValueError:
        canon_scheme = final_scheme = ""
    if canon_scheme == "http" and final_scheme == "https":
        return [
            _result(
                "canonical",
                title,
                "warn",
                0.7,
                4,
                "canonical이 보안 연결이 없는 http:// 주소를 가리킵니다.",
                f'canonical을 https 주소로 고치세요: <link rel="canonical" href="{clean}">',
                evidence,
            )
        ]
    detail = "대표 주소가 이 페이지로 올바르게 지정되어 있습니다."
    if canon_host != final_host:
        detail += f" (www 유무가 다릅니다: {canon_host}. 접속 주소도 대표 주소로 301 리다이렉트되게 하면 더 좋습니다.)"
    return [_result("canonical", title, "pass", 1.0, 4, detail, None, evidence)]


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------


def run(snapshot: SiteSnapshot, page: ParsedPage) -> list[CheckResult]:
    results: list[CheckResult] = []
    results += _safe("http_status", "페이지 정상 응답 (HTTP 200)", lambda: _check_http_status(snapshot))
    results += _safe("https", "HTTPS 보안 연결", lambda: _check_https(snapshot))

    if snapshot.fetch_error:
        results.append(
            _result(
                "skipped",
                "나머지 기술 항목",
                "info",
                0.0,
                0.0,
                "페이지를 불러오지 못해 robots.txt, 색인 지시어, llms.txt 등 나머지 기술 항목은 확인하지 못했습니다.",
                "페이지 접속 문제를 해결한 뒤 다시 진단하세요.",
            )
        )
        return results

    path = _robots_path(snapshot.final_url or snapshot.url)
    results += _safe("ai_search_bots", "AI 검색·답변 크롤러 접근 (robots.txt)", lambda: _check_search_bots(snapshot, path))
    results += _safe("ai_training_bots", "AI 모델 학습용 크롤러 정책 (참고)", lambda: _check_training_bots(snapshot, path))
    results += _safe("indexing", "검색 색인 허용 (noindex 없음)", lambda: _check_indexing(snapshot, page))
    results += _safe("server_rendered_text", "자바스크립트 없이 읽히는 본문", lambda: _check_rendered_text(snapshot, page))
    results += _safe("llms_txt", "llms.txt (선택 사항)", lambda: _check_llms_txt(snapshot))
    results += _safe("sitemap", "사이트맵", lambda: _check_sitemap(snapshot))
    results += _safe("html_lang", "페이지 언어 표시 (html lang)", lambda: _check_lang(page))
    results += _safe("canonical", "대표 주소 (canonical)", lambda: _check_canonical(snapshot, page))
    return results
