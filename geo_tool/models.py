"""모든 모듈이 공유하는 데이터 구조."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Category = Literal["technical", "structure", "schema", "content", "safety"]
Status = Literal["pass", "warn", "fail", "info"]

CATEGORY_LABELS: dict[str, str] = {
    "technical": "기술 접근성",
    "structure": "콘텐츠 구조",
    "schema": "구조화 데이터",
    "content": "인용 가능성",
    "safety": "위험·준법",
}


@dataclass
class SiteSnapshot:
    """수집 단계 결과. 네트워크 접근은 fetch.py에서만 일어난다."""

    url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    html: str
    robots_txt: str | None = None
    llms_txt: str | None = None
    sitemap_found: bool = False
    fetch_error: str | None = None


@dataclass
class CheckResult:
    """진단 항목 하나의 결과.

    score는 0~1 사이 획득 비율, weight는 항목 중요도(배점)다.
    status가 "info"인 항목은 점수 계산에서 제외된다.
    """

    id: str
    category: Category
    title: str
    status: Status
    score: float
    weight: float
    detail: str
    fix: str | None = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FixFile:
    """사이트에 그대로 적용할 수 있는 생성 결과물."""

    filename: str
    title: str
    description: str
    content: str
    language: str = "text"


@dataclass
class CategoryScore:
    category: str
    label: str
    score: float  # 0~100
    earned: float
    possible: float


@dataclass
class ScoreSummary:
    total: float  # 0~100
    grade: str  # A~F
    categories: list[CategoryScore]


@dataclass
class AuditResult:
    url: str
    final_url: str
    fetched_at: str
    summary: ScoreSummary
    checks: list[CheckResult]
    fixes: list[FixFile]
    ai: dict | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
