"""Get text in from files, directories and URLs.

The URL loader needs requests and BeautifulSoup, which are optional, so the
import happens inside the function. `html_to_text` is pure and testable without
either of them.
"""

import os
import re
from typing import List, Optional

from .types import Document

_SCRIPT = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")


def html_to_text(html: str) -> str:
    """Strip scripts, styles and tags, and collapse the leftover whitespace.

    Deliberately simple. A reader mode that keeps headings and lists would be
    better for retrieval, since those carry structure the model can use, but this
    is enough to turn a page into prose.
    """
    text = _SCRIPT.sub(" ", html or "")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = _TAG.sub(" ", text)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def load_text(path: str, encoding: str = "utf-8") -> Document:
    with open(path, encoding=encoding, errors="replace") as handle:
        text = handle.read()
    return Document(text=text, source=os.path.basename(path), metadata={"path": path})


def load_directory(path: str, glob: str = "*.md", recursive: bool = False) -> List[Document]:
    """Load every matching file. Skips anything unreadable rather than dying."""
    from glob import glob as _glob

    pattern = os.path.join(path, "**", glob) if recursive else os.path.join(path, glob)
    documents: List[Document] = []
    for match in sorted(_glob(pattern, recursive=recursive)):
        if not os.path.isfile(match):
            continue
        try:
            documents.append(load_text(match))
        except OSError:
            continue
    return documents


def load_url(url: str, timeout: int = 30, user_agent: Optional[str] = None) -> Document:
    """Fetch a page and reduce it to text."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "load_url needs requests and beautifulsoup4. "
            "Install the optional stack with: pip install -r requirements.txt"
        ) from exc

    headers = {"User-Agent": user_agent} if user_agent else None
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = _BLANK_LINES.sub("\n\n", _WHITESPACE.sub(" ", text).strip())
    return Document(text=text, source=url, metadata={"url": url, "status": response.status_code})
