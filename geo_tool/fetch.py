"""대상 사이트 수집. 이 모듈만 네트워크에 접근한다."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

from . import __version__
from .models import SiteSnapshot

USER_AGENT = f"Mozilla/5.0 (compatible; GEO-Tool/{__version__}; +https://github.com/noob2220202/geo)"
TIMEOUT = 15
MAX_BYTES = 5 * 1024 * 1024


class FetchError(Exception):
    pass


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise FetchError("URL을 입력해 주세요.")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise FetchError("http:// 또는 https:// 주소만 진단할 수 있습니다.")
    return url


def _is_private_host(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False  # 해석 실패는 요청 단계에서 오류로 드러난다
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def _get(session: requests.Session, url: str) -> requests.Response:
    resp = session.get(url, timeout=TIMEOUT, stream=True, allow_redirects=True)
    chunks, size = [], 0
    for chunk in resp.iter_content(64 * 1024):
        size += len(chunk)
        if size > MAX_BYTES:
            resp.close()
            raise FetchError(f"페이지가 너무 큽니다 (최대 {MAX_BYTES // 1024 // 1024}MB).")
        chunks.append(chunk)
    resp._content = b"".join(chunks)
    return resp


def _get_text(session: requests.Session, url: str) -> str | None:
    """robots.txt, llms.txt 같은 보조 파일. 없거나 HTML 오류 페이지면 None."""
    try:
        resp = _get(session, url)
    except (requests.RequestException, FetchError):
        return None
    if resp.status_code != 200:
        return None
    ctype = resp.headers.get("Content-Type", "")
    text = resp.text
    if "html" in ctype.lower() or text.lstrip().lower().startswith(("<!doctype html", "<html")):
        return None
    return text


def fetch_site(url: str, *, allow_private: bool = False) -> SiteSnapshot:
    url = normalize_url(url)
    host = urlparse(url).hostname or ""
    if not allow_private and _is_private_host(host):
        raise FetchError("내부망/로컬 주소는 진단할 수 없습니다.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ko,en;q=0.8"})

    try:
        resp = _get(session, url)
    except requests.RequestException as e:
        return SiteSnapshot(url=url, final_url=url, status_code=0, headers={}, html="", fetch_error=f"접속 실패: {e}")

    final_url = resp.url
    final_host = urlparse(final_url).hostname or ""
    if not allow_private and final_host != host and _is_private_host(final_host):
        raise FetchError("내부망/로컬 주소로 리다이렉트되어 진단을 중단했습니다.")

    root = f"{urlparse(final_url).scheme}://{urlparse(final_url).netloc}/"
    robots = _get_text(session, urljoin(root, "robots.txt"))
    llms = _get_text(session, urljoin(root, "llms.txt"))

    sitemap_found = False
    if robots and "sitemap:" in robots.lower():
        sitemap_found = True
    else:
        sitemap_found = _get_text(session, urljoin(root, "sitemap.xml")) is not None

    resp.encoding = resp.encoding or resp.apparent_encoding
    return SiteSnapshot(
        url=url,
        final_url=final_url,
        status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items()},
        html=resp.text,
        robots_txt=robots,
        llms_txt=llms,
        sitemap_found=sitemap_found,
    )
