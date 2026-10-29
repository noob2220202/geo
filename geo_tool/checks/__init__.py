"""진단 모듈 모음.

각 모듈은 `run(snapshot: SiteSnapshot, page: ParsedPage) -> list[CheckResult]`를 제공한다.
네트워크 접근 없이 순수 함수로 동작해야 한다.
"""

from __future__ import annotations

from ..models import CheckResult, SiteSnapshot
from ..page import ParsedPage
from . import content, safety, schema, structure, technical

MODULES = [technical, structure, schema, content, safety]


def run_all(snapshot: SiteSnapshot, page: ParsedPage) -> list[CheckResult]:
    results: list[CheckResult] = []
    for module in MODULES:
        results.extend(module.run(snapshot, page))
    return results
