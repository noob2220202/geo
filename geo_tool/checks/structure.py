"""콘텐츠 구조 진단.

제목·설명·소제목·목록·문단·분량처럼 AI 검색엔진이 페이지 내용을 잘라 인용하기 쉬운
구조인지 살핀다. 네트워크 접근 없이 ParsedPage만 본다.
"""

from __future__ import annotations

import re
from statistics import mean

from bs4 import Tag

from ..models import CheckResult, SiteSnapshot, Status
from ..page import ParsedPage

CATEGORY = "structure"

TITLE_MIN, TITLE_MAX = 10, 65
DESC_MIN, DESC_MIN_KO, DESC_MAX = 50, 40, 160
THIN_FAIL, THIN_WARN = 300, 1500
THIN_WARN_WORDS_EN = 300
WALL_CHARS = 600

_WS = re.compile(r"\s+")

_GENERIC_TITLES = {
    "home", "homepage", "main", "index", "untitled", "welcome", "new page", "document",
    "홈", "홈페이지", "메인", "메인페이지", "메인 페이지", "제목 없음", "환영합니다",
}

# 메뉴·바닥글처럼 본문이 아닌 영역 (식당 '메뉴'와 헷갈리지 않도록 menu는 넣지 않는다)
_CHROME_TAGS = {"nav", "header", "footer", "aside"}
_CHROME_ROLES = {"navigation", "menubar", "banner", "contentinfo"}
_CHROME_HINT = re.compile(r"(?:^|[\s_-])(?:nav|navbar|navigation|gnb|lnb|snb|footer|breadcrumbs?)(?:[\s_-]|$)", re.I)

_EN_QUESTION = re.compile(
    r"^(?:what|how|why|when|where|which|who|can|is|are|does|should)\b", re.I
)
_Q_PREFIX = re.compile(r"^(?:q\d*|질문\s*\d*)\s*[.:)\]]", re.I)
_KO_ENDINGS = ("까요", "나요", "니까", "는가", "은가", "인가", "한가", "는지")
_TRAILING = " \t.!。…~)]}\"'”’»"

_POOR_ALT = re.compile(
    r"^(?:image|img|photo|picture|pic|banner|thumbnail|thumb|이미지|사진|그림|배너|썸네일)\s*\d*$"
    r"|\.(?:jpe?g|png|gif|webp|svg|avif|bmp)$"
    r"|^(?:img|dsc|image|photo)[_-]?\d+",
    re.I,
)


# ---------------------------------------------------------------- 공용 도우미

def _clean(text: str | None) -> str:
    return _WS.sub(" ", text or "").strip()


def _short(text: str, n: int = 60) -> str:
    text = _clean(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _evidence(items: list[str] | None) -> list[str]:
    out: list[str] = []
    for item in items or []:
        item = _clean(item)
        if item:
            out.append(item if len(item) <= 200 else item[:199] + "…")
        if len(out) >= 5:
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
    if status == "info":
        score, weight = 0.0, 0.0
    return CheckResult(
        id=f"{CATEGORY}.{name}",
        category=CATEGORY,
        title=title,
        status=status,
        score=round(max(0.0, min(1.0, float(score))), 3),
        weight=float(weight),
        detail=detail,
        fix=fix,
        evidence=_evidence(evidence),
    )


def _hangul_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "가" <= c <= "힣") / len(letters)


def _jongseong(ch: str) -> int:
    """한글 음절의 받침 번호(없으면 0). 한글 음절이 아니면 -1."""
    code = ord(ch) - 0xAC00
    if 0 <= code < 11172:
        return code % 28
    return -1


def _body(page: ParsedPage):
    return page.soup.body or page.soup


def _in_chrome(tag: Tag) -> bool:
    """메뉴·머리글·바닥글 등 본문이 아닌 영역 안에 있는지."""
    node = tag
    depth = 0
    while isinstance(node, Tag) and depth < 80:
        if node.name in _CHROME_TAGS:
            return True
        role = node.get("role")
        if isinstance(role, list):
            role = " ".join(role)
        if isinstance(role, str) and role.strip().lower() in _CHROME_ROLES:
            return True
        cls = node.get("class") or []
        ident = " ".join(cls if isinstance(cls, list) else [str(cls)])
        node_id = node.get("id")
        if isinstance(node_id, str):
            ident += " " + node_id
        if ident and _CHROME_HINT.search(ident):
            return True
        node = node.parent
        depth += 1
    return False


def is_question_heading(text: str) -> bool:
    """사용자 질문 형태의 소제목인지 (한국어 의문형 어미·의문사, 영어 의문사, 물음표)."""
    t = _clean(text)
    if not t:
        return False
    if "?" in t or "？" in t:
        return True
    if _Q_PREFIX.match(t) or _EN_QUESTION.match(t):
        return True

    tokens = [tok.strip(_TRAILING + ",:;") for tok in t.split(" ")]
    tokens = [tok for tok in tokens if tok]
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok.startswith(("무엇", "어떻게", "뭔가요", "뭘까")):
            return True
        if tok in ("왜", "왜요", "왜죠"):
            return True
        if tok.startswith("언제") and not tok.startswith(("언제나", "언제든")):
            return True
        if tok.startswith("어디") and not tok.startswith(("어디서나", "어디든", "어디나")):
            return True
        if tok.startswith("얼마") and tok != "얼마간":
            if tok == "얼마" and nxt.startswith(("전", "후", "뒤", "동안", "안", "지나")):
                continue
            return True

    core = t.rstrip(_TRAILING)
    if core.endswith(_KO_ENDINGS):
        return True
    # "필요한가요", "좋은가요": 'ㄴ' 받침 + 가요 / "할까", "될까": 'ㄹ' 받침 + 까
    if len(core) >= 3 and core.endswith("가요") and _jongseong(core[-3]) == 4:
        return True
    if len(core) >= 2 and core.endswith("까") and _jongseong(core[-2]) == 8:
        return True
    return False


# ---------------------------------------------------------------- 개별 진단

def _check_title(page: ParsedPage) -> CheckResult:
    name, label, weight = "title", "페이지 제목(title)", 8
    title = _clean(page.title)
    n = len(title)
    example = "예: <title>강남역 임플란트 비용과 치료 기간 안내 | ○○치과</title>"
    if not title:
        return _result(
            name, label, "fail", 0, weight,
            "<title> 태그가 없거나 비어 있습니다. 검색엔진과 AI 답변은 이 제목으로 페이지 주제를 가장 먼저 파악하고, 출처를 표시할 때도 사용합니다.",
            fix=f"<head> 안에 페이지마다 다른 <title>을 넣으세요. '핵심 주제 + 브랜드명' 형식이 좋습니다. {example}",
        )
    if title.lower() in _GENERIC_TITLES:
        return _result(
            name, label, "warn", 0.3, weight,
            f"제목이 '{title}'처럼 일반적인 단어라 어떤 페이지인지 알 수 없습니다.",
            fix=f"이 페이지가 다루는 주제와 브랜드명을 제목에 넣으세요. {example}",
            evidence=[title],
        )
    if n < TITLE_MIN:
        return _result(
            name, label, "warn", 0.5, weight,
            f"제목이 {n}자로 짧습니다. 주제를 충분히 설명하지 못하면 AI가 어떤 질문에 이 페이지를 인용할지 판단하기 어렵습니다.",
            fix=f"제목을 {TITLE_MIN}~{TITLE_MAX}자 정도로 늘려 핵심 키워드와 브랜드명을 함께 담으세요. {example}",
            evidence=[title],
        )
    if n > TITLE_MAX:
        return _result(
            name, label, "warn", 0.6, weight,
            f"제목이 {n}자로 깁니다. 검색 결과나 AI 답변의 출처 표시에서 뒷부분이 잘릴 수 있습니다.",
            fix=f"가장 중요한 키워드를 앞쪽에 두고 {TITLE_MAX}자 안쪽으로 줄이세요.",
            evidence=[title],
        )
    return _result(name, label, "pass", 1, weight, f"제목이 {n}자로 적절합니다.", evidence=[title])


def _check_meta_description(page: ParsedPage) -> CheckResult:
    name, label, weight = "meta_description", "메타 설명(description)", 6
    desc = _clean(page.meta_description)
    n = len(desc)
    min_len = DESC_MIN_KO if _hangul_ratio(desc) >= 0.3 else DESC_MIN
    tag_example = '<meta name="description" content="이 페이지가 무엇을, 누구에게, 왜 도움이 되는지 한두 문장으로 요약">'
    if not desc:
        return _result(
            name, label, "fail", 0, weight,
            "메타 설명이 없습니다. 검색엔진과 AI 도구가 페이지를 요약하거나 미리보기를 만들 때 이 문장을 참고합니다.",
            fix=f"<head> 안에 {min_len}~{DESC_MAX}자 정도의 요약 문장을 넣으세요. {tag_example}",
        )
    if n < min_len:
        return _result(
            name, label, "warn", 0.5, weight,
            f"메타 설명이 {n}자로 짧습니다.",
            fix=f"핵심 내용·대상·차별점을 담아 {min_len}~{DESC_MAX}자 정도로 보강하세요.",
            evidence=[desc],
        )
    if n > DESC_MAX:
        return _result(
            name, label, "warn", 0.7, weight,
            f"메타 설명이 {n}자로 깁니다. 미리보기에서 잘릴 수 있습니다.",
            fix=f"가장 중요한 내용을 앞에 두고 {DESC_MAX}자 안쪽으로 줄이세요.",
            evidence=[desc],
        )
    return _result(name, label, "pass", 1, weight, f"메타 설명이 {n}자로 적절합니다.", evidence=[desc])


def _check_h1(page: ParsedPage) -> CheckResult:
    name, label, weight = "h1", "대표 제목(H1)", 7
    h1s = [text for level, text in page.headings if level == 1]
    if not h1s:
        return _result(
            name, label, "fail", 0, weight,
            "H1 태그가 없습니다. H1은 페이지의 대표 제목으로, AI가 본문의 주제를 파악하는 기준이 됩니다.",
            fix="본문 맨 위의 대표 제목을 <h1>로 감싸세요. 로고나 메뉴가 아니라 이 페이지의 주제를 담은 문장이어야 합니다. 예: <h1>임플란트 비용과 치료 기간 안내</h1>",
        )
    if len(h1s) > 1:
        return _result(
            name, label, "warn", 0.5, weight,
            f"H1이 {len(h1s)}개 있습니다. 대표 제목이 여러 개면 페이지의 중심 주제가 흐려집니다.",
            fix="대표 제목 하나만 <h1>로 두고 나머지는 <h2>·<h3> 소제목으로 바꾸세요.",
            evidence=[f"H1: {_short(t, 120)}" for t in h1s],
        )
    if not h1s[0]:
        return _result(
            name, label, "warn", 0.4, weight,
            "H1이 있지만 글자가 없습니다(이미지 로고만 들어 있는 경우 등).",
            fix="H1 안에 페이지 주제를 설명하는 텍스트를 넣으세요. 로고 이미지는 H1 밖으로 빼는 것이 좋습니다.",
        )
    return _result(name, label, "pass", 1, weight, "H1이 하나 있습니다.", evidence=[f"H1: {_short(h1s[0], 120)}"])


def _check_heading_hierarchy(page: ParsedPage) -> CheckResult:
    name, label, weight = "heading_hierarchy", "소제목 계층", 4
    headings = page.headings
    if len(headings) < 2:
        return _result(
            name, label, "info", 0, 0,
            "소제목이 거의 없어 계층 검사를 생략했습니다. (소제목 개수는 '본문 섹션 나누기' 항목에서 따로 봅니다.)",
        )
    skips: list[str] = []
    for (prev_level, prev_text), (level, text) in zip(headings, headings[1:]):
        if level > prev_level + 1:
            skips.append(f"H{prev_level} '{_short(prev_text, 40)}' → H{level} '{_short(text, 40)}'")
    if not skips:
        return _result(name, label, "pass", 1, weight, "소제목 단계(H1→H2→H3)가 건너뛰지 않고 이어집니다.")
    score = 0.6 if len(skips) == 1 else max(0.2, 1 - 0.2 * len(skips))
    return _result(
        name, label, "warn", score, weight,
        f"소제목 단계를 건너뛴 곳이 {len(skips)}군데 있습니다. 단계가 어긋나면 AI가 어떤 내용이 어느 주제에 속하는지 잘못 읽을 수 있습니다.",
        fix="소제목은 H1 → H2 → H3 순서로 한 단계씩 내려가게 쓰세요. 글자 크기를 바꾸려고 태그 단계를 고르지 말고, 크기는 CSS로 조정하세요.",
        evidence=skips,
    )


def _check_sections(page: ParsedPage) -> CheckResult:
    name, label, weight = "sections", "본문 섹션 나누기", 6
    subs = [text for level, text in page.headings if level in (2, 3)]
    n = len(subs)
    if page.char_count >= THIN_WARN:
        required = 3
    elif page.char_count >= 500:
        required = 2
    else:
        required = 1
    fix = (
        "주제별로 문단을 묶고 각 묶음 앞에 <h2> 소제목을 붙이세요. 고객이 실제로 묻는 말을 소제목으로 쓰면 좋습니다. "
        "예: <h2>비용은 얼마인가요?</h2>, <h2>진행 절차</h2>, <h2>주의할 점</h2>"
    )
    evidence = [_short(t, 80) for t in subs]
    if n >= required:
        return _result(name, label, "pass", 1, weight, f"H2·H3 소제목이 {n}개 있어 내용이 섹션으로 나뉘어 있습니다.", evidence=evidence)
    if n == 0:
        return _result(
            name, label, "fail", 0, weight,
            "H2·H3 소제목이 없습니다. AI는 소제목 단위로 내용을 잘라 질문에 맞는 부분을 인용하므로, 섹션이 없으면 인용하기 어렵습니다.",
            fix=fix,
        )
    return _result(
        name, label, "warn", n / required, weight,
        f"H2·H3 소제목이 {n}개뿐입니다. 이 정도 분량이면 {required}개 이상으로 나누는 것이 좋습니다.",
        fix=fix,
        evidence=evidence,
    )


def _check_question_headings(page: ParsedPage) -> CheckResult:
    name, label, weight = "question_headings", "질문형 소제목", 5
    texts = [text for _, text in page.headings if text]
    questions = [t for t in texts if is_question_heading(t)]
    q = len(questions)
    fix = (
        "고객이 실제로 검색하거나 AI에게 물어볼 질문을 소제목으로 쓰고, 바로 아래 첫 문장에서 답을 먼저 말하세요. "
        "예: <h2>GEO는 무엇인가요?</h2> <p>GEO는 ChatGPT 같은 AI 검색 답변에 내 사이트가 출처로 쓰이기 쉽게 만드는 작업입니다.</p>"
    )
    if not texts:
        return _result(
            name, label, "warn", 0, weight,
            "소제목이 없어 질문형 구조를 찾을 수 없습니다. AI 답변은 사용자 질문과 비슷한 소제목 아래의 내용을 즐겨 인용합니다.",
            fix=fix,
        )
    evidence = [_short(t, 100) for t in questions]
    if q >= 2 or (q == 1 and len(texts) <= 2):
        return _result(name, label, "pass", 1, weight, f"질문 형태의 소제목이 {q}개 있습니다.", evidence=evidence)
    if q == 1:
        return _result(
            name, label, "warn", 0.6, weight,
            "질문 형태의 소제목이 1개 있습니다. 질문형 소제목이 더 있으면 AI가 사용자 질문과 페이지 섹션을 연결하기 쉬워집니다.",
            fix=fix,
            evidence=evidence,
        )
    return _result(
        name, label, "warn", 0, weight,
        f"소제목 {len(texts)}개 중 질문 형태가 없습니다. AI 답변은 사용자 질문과 비슷한 소제목 아래의 내용을 즐겨 인용합니다.",
        fix=fix,
        evidence=[_short(t, 80) for t in texts[:3]],
    )


def _check_lists_tables(page: ParsedPage) -> CheckResult:
    name, label, weight = "lists_tables", "목록·표 활용", 4
    body = _body(page)
    lists = tables = dls = chrome_lists = 0
    for tag in body.find_all(["ul", "ol", "table", "dl"]):
        if tag.name in ("ul", "ol"):
            if len(tag.find_all("li")) < 2:
                continue
            if _in_chrome(tag):
                chrome_lists += 1
            else:
                lists += 1
        elif tag.name == "table":
            role = tag.get("role")
            if role in ("presentation", "none") or tag.find("table") is not None:
                continue  # 레이아웃용 표
            if len(tag.find_all("tr")) >= 2 and not _in_chrome(tag):
                tables += 1
        elif tag.find("dt") is not None and tag.find("dd") is not None and not _in_chrome(tag):
            dls += 1

    found = lists + tables + dls
    parts = []
    if lists:
        parts.append(f"목록(ul/ol) {lists}개")
    if tables:
        parts.append(f"표 {tables}개")
    if dls:
        parts.append(f"정의 목록(dl) {dls}개")
    fix = (
        "절차·특징·가격·비교처럼 나열되는 정보는 문장 대신 <ul>/<ol> 목록이나 <table> 표로 정리하세요. "
        "AI는 목록과 표를 그대로 뽑아 답변에 쓰기 쉽습니다. 예: <ol><li>상담 예약</li><li>정밀 검사</li><li>치료 시작</li></ol>"
    )
    if found:
        return _result(name, label, "pass", 1, weight, f"본문에 {', '.join(parts)}가 있습니다.", evidence=parts)
    if chrome_lists:
        return _result(
            name, label, "warn", 0.2, weight,
            "메뉴·바닥글에만 목록이 있고 본문에는 목록이나 표가 없습니다.",
            fix=fix,
        )
    return _result(name, label, "warn", 0, weight, "본문에 목록이나 표가 없습니다.", fix=fix)


def _check_paragraphs(page: ParsedPage) -> CheckResult:
    name, label, weight = "paragraphs", "문단 길이(읽기 쉬움)", 4
    body = _body(page)
    paras = [_clean(p.get_text(" ")) for p in body.find_all("p") if not _in_chrome(p)]
    paras = [p for p in paras if p]
    if not paras:
        if page.char_count >= THIN_FAIL:
            return _result(
                name, label, "info", 0, 0,
                "본문이 <p> 문단 태그 없이 구성되어 있어 문단 길이를 평가하지 않았습니다.",
                fix="가능하면 본문 문장을 <p> 문단으로 나눠 주세요. 문단 경계가 분명하면 AI가 필요한 부분만 정확히 인용하기 쉽습니다.",
            )
        return _result(name, label, "info", 0, 0, "평가할 문단이 없습니다.")

    lengths = [len(p) for p in paras]
    avg = mean(lengths)
    walls = [p for p in paras if len(p) > WALL_CHARS]
    longest = max(paras, key=len)
    evidence = [f"문단 {len(paras)}개, 평균 {avg:.0f}자", f"가장 긴 문단({len(longest)}자): {_short(longest, 120)}"]
    fix = (
        f"한 문단에는 한 가지 내용만 담고 3~5문장(대략 {WALL_CHARS // 2}자 안팎)에서 끊으세요. "
        "문단 첫 문장에 핵심 결론을 먼저 쓰면 AI가 그 문장을 그대로 인용하기 좋습니다."
    )
    if avg > WALL_CHARS or len(walls) >= 3:
        return _result(
            name, label, "warn", 0.4, weight,
            f"긴 문단이 많습니다(평균 {avg:.0f}자, {WALL_CHARS}자 넘는 문단 {len(walls)}개). 글이 한 덩어리로 뭉쳐 있으면 AI가 필요한 문장을 골라내기 어렵습니다.",
            fix=fix,
            evidence=evidence,
        )
    if walls:
        return _result(
            name, label, "warn", 0.75, weight,
            f"{WALL_CHARS}자가 넘는 긴 문단이 {len(walls)}개 있습니다.",
            fix=fix,
            evidence=evidence,
        )
    return _result(name, label, "pass", 1, weight, f"문단 길이가 적당합니다(평균 {avg:.0f}자).", evidence=evidence[:1])


def _check_content_depth(page: ParsedPage) -> CheckResult:
    name, label, weight = "content_depth", "본문 분량", 7
    chars, words = page.char_count, page.word_count
    stats = f"본문 글자 수(공백 제외) {chars:,}자, 단어 {words:,}개"
    english = _hangul_ratio(page.text[:5000]) < 0.3
    fix = (
        "고객이 자주 묻는 질문, 구체적인 수치(가격·기간·수량), 절차, 사례, 주의사항을 본문 텍스트로 추가하세요. "
        "이미지 속 글자나 자바스크립트로만 그려지는 내용은 AI가 읽지 못할 수 있으니 HTML 텍스트로 넣어 주세요."
    )
    if chars < THIN_FAIL:
        return _result(
            name, label, "fail", 0.1 if chars else 0, weight,
            f"{stats}로 내용이 거의 없습니다. AI가 인용할 만한 정보가 부족합니다.",
            fix=fix,
            evidence=[stats],
        )
    if chars < THIN_WARN and not (english and words >= THIN_WARN_WORDS_EN):
        score = 0.3 + 0.5 * (chars - THIN_FAIL) / (THIN_WARN - THIN_FAIL)
        return _result(
            name, label, "warn", score, weight,
            f"{stats}로 다소 얇은 편입니다. 질문에 답할 만큼 구체적인 정보가 있어야 AI 답변에 인용될 가능성이 높아집니다.",
            fix=fix,
            evidence=[stats],
        )
    return _result(name, label, "pass", 1, weight, f"{stats}로 충분한 분량입니다.", evidence=[stats])


def _check_image_alt(page: ParsedPage) -> CheckResult:
    name, label, weight = "image_alt", "이미지 대체 텍스트(alt)", 3
    body = _body(page)
    total = good = empty = 0
    missing: list[str] = []
    for img in body.find_all("img"):
        if str(img.get("width", "")).strip() in ("0", "1") or str(img.get("height", "")).strip() in ("0", "1"):
            continue  # 추적용 픽셀
        src = img.get("src") or img.get("data-src") or ""
        if not isinstance(src, str):
            src = ""
        total += 1
        alt = img.get("alt")
        if alt is None:
            missing.append(f"alt 없음: {_short(src, 150) or '(src 없음)'}")
        elif not alt.strip():
            empty += 1
        elif _POOR_ALT.search(alt.strip()):
            missing.append(f"의미 없는 alt '{_short(alt, 40)}': {_short(src, 120)}")
        else:
            good += 1
    if total == 0:
        return _result(name, label, "info", 0, 0, "본문에 이미지가 없습니다.")
    coverage = (good + 0.5 * empty) / total
    detail = f"이미지 {total}개 중 설명이 있는 것 {good}개, 빈 alt {empty}개, 없거나 의미 없는 것 {len(missing)}개."
    fix = (
        "정보를 담은 이미지에는 무엇을 보여주는지 구체적으로 쓴 alt를 넣으세요. 예: <img src=\"menu.jpg\" alt=\"점심 정식 메뉴판 – 김치찌개 9,000원\">. "
        "순수 장식용 이미지만 alt=\"\"로 두세요. 가격·일정처럼 중요한 정보는 이미지에만 두지 말고 본문 텍스트로도 쓰세요."
    )
    if coverage >= 0.9:
        return _result(name, label, "pass", coverage, weight, detail)
    status: Status = "warn" if coverage >= 0.5 else "fail"
    return _result(name, label, status, coverage, weight, detail, fix=fix, evidence=missing)


def _check_open_graph(page: ParsedPage) -> CheckResult:
    name, label, weight = "open_graph", "오픈그래프(공유 미리보기)", 2
    keys = ("og:title", "og:description", "og:image")
    present = [k for k in keys if page.og.get(k)]
    absent = [k for k in keys if k not in present]
    evidence = [f"{k}: {_short(page.og[k], 120)}" for k in present]
    if not absent:
        return _result(name, label, "pass", 1, weight, "og:title, og:description, og:image가 모두 있습니다.", evidence=evidence)
    tags = " ".join(f'<meta property="{k}" content="...">' for k in absent)
    return _result(
        name, label, "warn", len(present) / len(keys), weight,
        f"오픈그래프 태그 중 {', '.join(absent)}가 없습니다. 링크 공유 미리보기와 일부 AI 도구가 이 정보로 페이지를 요약합니다.",
        fix=f"<head>에 다음 태그를 추가하세요: {tags}",
        evidence=evidence,
    )


def run(snapshot: SiteSnapshot, page: ParsedPage) -> list[CheckResult]:
    if snapshot.fetch_error and not (page.raw_html or "").strip():
        return [
            _result(
                "skipped", "콘텐츠 구조 진단", "info", 0, 0,
                f"페이지를 가져오지 못해 콘텐츠 구조 진단을 건너뛰었습니다. ({snapshot.fetch_error})",
            )
        ]
    return [
        _check_title(page),
        _check_meta_description(page),
        _check_h1(page),
        _check_heading_hierarchy(page),
        _check_sections(page),
        _check_question_headings(page),
        _check_lists_tables(page),
        _check_paragraphs(page),
        _check_content_depth(page),
        _check_image_alt(page),
        _check_open_graph(page),
    ]
