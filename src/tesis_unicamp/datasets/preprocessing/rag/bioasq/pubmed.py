import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

from tesis_unicamp.datasets.preprocessing.rag.bioasq.constants import (
    PUBMED_BATCH_SIZE,
    PUBMED_EFETCH_URL,
    PUBMED_MAX_RETRIES,
    PUBMED_REQUEST_DELAY_SECONDS,
)

PMID_PATTERN = re.compile(r"/(\d+)/?$")


def extract_pmid(document_url: str) -> str | None:
    """Extract PubMed ID from a BioASQ document URL."""
    match = PMID_PATTERN.search(document_url.strip())
    return match.group(1) if match else None


def _parse_pubmed_xml(xml_text: str) -> dict[str, dict[str, str]]:
    root = ET.fromstring(xml_text)
    records: dict[str, dict[str, str]] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue

        pmid = pmid_el.text.strip()
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        abstract_parts: list[str] = []
        for abstract_text in article.findall(".//AbstractText"):
            label = abstract_text.get("Label")
            text = "".join(abstract_text.itertext()).strip()
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)

        records[pmid] = {
            "title": title,
            "text": " ".join(abstract_parts),
        }

    return records


def _fetch_pubmed_batch(pmids: list[str], email: str | None = None) -> dict[str, dict[str, str]]:
    """Fetch a batch of PubMed records using POST to avoid URL length limits."""
    data: dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        data["email"] = email

    response = requests.post(PUBMED_EFETCH_URL, data=data, timeout=120)
    response.raise_for_status()
    return _parse_pubmed_xml(response.text)


def _fetch_pubmed_batch_with_retry(
    pmids: list[str],
    email: str | None = None,
) -> dict[str, dict[str, str]]:
    if not pmids:
        return {}

    for attempt in range(PUBMED_MAX_RETRIES):
        try:
            return _fetch_pubmed_batch(pmids, email=email)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in {400, 414, 429, 500, 502, 503, 504}:
                raise
            if len(pmids) == 1:
                return {pmids[0]: {"title": "", "text": ""}}
            if attempt == PUBMED_MAX_RETRIES - 1:
                mid = len(pmids) // 2
                left = _fetch_pubmed_batch_with_retry(pmids[:mid], email=email)
                time.sleep(PUBMED_REQUEST_DELAY_SECONDS)
                right = _fetch_pubmed_batch_with_retry(pmids[mid:], email=email)
                left.update(right)
                return left
            time.sleep(PUBMED_REQUEST_DELAY_SECONDS * (attempt + 1))

    return {}


def load_pubmed_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    if not cache_path.exists():
        return {}
    with cache_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_pubmed_cache(cache_path: Path, cache: dict[str, dict[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)


def fetch_pubmed_abstracts(
    pmids: set[str],
    cache_dir: Path,
    email: str | None = None,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch PubMed titles and abstracts, using a local JSON cache."""
    cache_path = cache_dir / "pubmed_abstracts.json"
    cache = {} if force_refresh else load_pubmed_cache(cache_path)

    missing = sorted(pmid for pmid in pmids if pmid not in cache)
    if not missing:
        return {pmid: cache[pmid] for pmid in pmids if pmid in cache}

    for start in tqdm(range(0, len(missing), PUBMED_BATCH_SIZE), desc="Fetching PubMed"):
        batch = missing[start : start + PUBMED_BATCH_SIZE]
        batch_records = _fetch_pubmed_batch_with_retry(batch, email=email)
        cache.update(batch_records)

        for pmid in batch:
            cache.setdefault(pmid, {"title": "", "text": ""})

        save_pubmed_cache(cache_path, cache)
        time.sleep(PUBMED_REQUEST_DELAY_SECONDS)

    return {pmid: cache[pmid] for pmid in pmids if pmid in cache}
