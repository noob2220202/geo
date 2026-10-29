"""HTML을 한 번만 파싱해서 모든 진단 모듈이 같이 쓰는 ParsedPage를 만든다."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

from bs4 import BeautifulSoup

_WS = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    return _WS.sub(" ", text or "").strip()


@dataclass
class ParsedPage:
    soup: BeautifulSoup
    base_url: str
    title: str | None
    meta_description: str | None
    lang: str | None
    canonical: str | None
    meta_robots: str | None
    headings: list[tuple[int, str]]
    text: str
    word_count: int
    char_count: int
    jsonld: list[dict]
    jsonld_errors: int
    og: dict[str, str]
    links: list[tuple[str, str]]
    images: list[tuple[str, str | None]]
    raw_html: str = field(repr=False, default="")

    def jsonld_types(self) -> set[str]:
        types: set[str] = set()
        for obj in self.jsonld:
            t = obj.get("@type")
            if isinstance(t, str):
                types.add(t)
            elif isinstance(t, list):
                types.update(x for x in t if isinstance(x, str))
        return types


def _flatten_jsonld(data, out: list[dict]) -> None:
    if isinstance(data, list):
        for item in data:
            _flatten_jsonld(item, out)
    elif isinstance(data, dict):
        out.append(data)
        if "@graph" in data:
            _flatten_jsonld(data["@graph"], out)


def _meta(soup: BeautifulSoup, *, name: str | None = None, prop: str | None = None) -> str | None:
    if name:
        tag = soup.find("meta", attrs={"name": re.compile(f"^{re.escape(name)}$", re.I)})
    else:
        tag = soup.find("meta", attrs={"property": re.compile(f"^{re.escape(prop or '')}$", re.I)})
    if tag and tag.get("content"):
        return _clean(tag["content"])
    return None


def parse_page(html: str, base_url: str) -> ParsedPage:
    soup = BeautifulSoup(html or "", "html.parser")

    jsonld: list[dict] = []
    jsonld_errors = 0
    for tag in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        try:
            _flatten_jsonld(json.loads(tag.string or tag.get_text() or ""), jsonld)
        except (json.JSONDecodeError, TypeError):
            jsonld_errors += 1

    title_tag = soup.find("title")
    html_tag = soup.find("html")
    canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in [r.lower() for r in (v if isinstance(v, list) else [v])])

    og: dict[str, str] = {}
    for tag in soup.find_all("meta", attrs={"property": re.compile(r"^og:", re.I)}):
        if tag.get("content"):
            og[tag["property"].lower()] = _clean(tag["content"])

    headings = [
        (int(h.name[1]), _clean(h.get_text(" ")))
        for h in soup.find_all(re.compile(r"^h[1-6]$"))
    ]

    links: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        links.append((urljoin(base_url, href), _clean(a.get_text(" "))))

    images = [(urljoin(base_url, img.get("src", "")), img.get("alt")) for img in soup.find_all("img")]

    # 보이는 텍스트 추출용 사본: 스크립트/스타일 등 제거
    text_soup = BeautifulSoup(html or "", "html.parser")
    for tag in text_soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    body = text_soup.body or text_soup
    text = _clean(body.get_text(" "))

    return ParsedPage(
        soup=soup,
        base_url=base_url,
        title=_clean(title_tag.get_text()) if title_tag else None,
        meta_description=_meta(soup, name="description"),
        lang=(html_tag.get("lang") or None) if html_tag else None,
        canonical=urljoin(base_url, canonical_tag["href"]) if canonical_tag and canonical_tag.get("href") else None,
        meta_robots=_meta(soup, name="robots"),
        headings=headings,
        text=text,
        word_count=len(text.split()) if text else 0,
        char_count=len(text.replace(" ", "")),
        jsonld=jsonld,
        jsonld_errors=jsonld_errors,
        og=og,
        links=links,
        images=images,
        raw_html=html or "",
    )
