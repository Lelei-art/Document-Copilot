"""Keyword extraction for Postgres full-text search."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval.types import SearchFilters

_FILLER_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "across",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "change",
        "changed",
        "describe",
        "describes",
        "described",
        "did",
        "do",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "its",
        "of",
        "on",
        "or",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "way",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)

_LOW_VALUE_FTS_WORDS = frozenset(
    {
        "driver",
        "drivers",
        "describe",
        "described",
        "describes",
        "change",
        "changed",
        "across",
        "way",
        "ks",
        "k",
    }
)

_KNOWN_PHRASES = (
    "customer concentration",
    "data center",
    "revenue mix",
    "cloud capacity",
    "ai infrastructure",
)

_TICKER_COMPANY_PREFIXES: dict[str, str] = {
    "AAPL": "apple",
    "AMZN": "amazon",
    "GOOGL": "google",
    "MSFT": "microsoft",
    "NVDA": "nvidia",
}


class FtsKeywordExtraction(BaseModel):
    terms: list[str] = Field(
        min_length=1,
        description="3-5 search terms for full-text search",
    )


def _token_count(query: str) -> int:
    return len(query.split())


def _normalize_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []

    for term in terms:
        cleaned = term.strip()
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        seen.add(key)
        normalized.append(cleaned)

    return normalized


def _is_company_token(token: str, filters: SearchFilters | None) -> bool:
    if filters is None or filters.ticker is None:
        return False

    prefix = _TICKER_COMPANY_PREFIXES.get(filters.ticker)
    if prefix is None:
        return False

    return token.casefold().startswith(prefix)


def _phrases_in_query(query: str) -> list[str]:
    lowered = query.casefold()
    found: list[str] = []

    for phrase in _KNOWN_PHRASES:
        start = lowered.find(phrase)
        if start == -1:
            continue
        found.append(query[start : start + len(phrase)])

    return found


def _capitalized_tokens(query: str, *, filters: SearchFilters | None) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:'s)?", query)
    caps: list[str] = []

    for token in tokens:
        bare = token.removesuffix("'s")

        if bare.casefold() in _FILLER_WORDS:
            continue
        if bare.casefold() in _LOW_VALUE_FTS_WORDS:
            continue
        if _is_company_token(token, filters):
            continue
        if len(bare) <= 2:
            continue

        if bare[0].isupper() or re.fullmatch(r"i[A-Z][A-Za-z0-9]*", bare):
            caps.append(bare)

    return caps


def _apply_word_budget(words: list[str]) -> list[str]:
    maximum = settings.retrieval_fts_keyword_max
    seen: set[str] = set()
    kept: list[str] = []

    for word in words:
        for part in word.split():
            key = part.casefold()

            if key in seen:
                continue
            if key in _FILLER_WORDS or key in _LOW_VALUE_FTS_WORDS:
                continue

            seen.add(key)
            kept.append(part)

            if len(kept) >= maximum:
                return kept

    return kept


def _flatten_term_list(terms: list[str]) -> list[str]:
    words: list[str] = []

    for term in terms:
        words.extend(term.split())

    return words


def _merge_fts_words(
    query: str,
    terms: list[str],
    *,
    filters: SearchFilters | None,
) -> list[str]:
    caps = _capitalized_tokens(query, filters=filters)
    phrases = _phrases_in_query(query)
    term_words = _apply_word_budget(_flatten_term_list(terms))

    candidates: list[str] = []

    if len(caps) >= 3:
        candidates.extend(caps)
    else:
        candidates.extend(phrases)
        candidates.extend(caps)

    candidates.extend(term_words)

    return _apply_word_budget(candidates)


def _deterministic_fallback(query: str, *, filters: SearchFilters | None) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/]*", query)
    kept: list[str] = []

    for token in tokens:
        if token.casefold() in _FILLER_WORDS:
            continue
        if _is_company_token(token, filters):
            continue
        kept.append(token)

    words = _merge_fts_words(query, kept, filters=filters)

    if not words:
        return query.strip()

    return " ".join(words)


def _build_user_message(query: str, filters: SearchFilters | None) -> str:
    parts = [f"Query: {query}"]

    if filters is not None and filters.ticker is not None:
        parts.append(f"Ticker filter: {filters.ticker} (omit company name from terms)")

    if filters is not None and filters.form is not None:
        parts.append(f"Form filter: {filters.form}")

    return "\n".join(parts)


def extract_fts_keywords(
    query: str,
    *,
    filters: SearchFilters | None = None,
) -> str:
    """Return a compact keyword string for plainto_tsquery without paid API calls."""

    stripped = query.strip()
    if not stripped:
        return stripped

    if _token_count(stripped) <= settings.retrieval_fts_keyword_fast_path_tokens:
        return stripped

    return _deterministic_fallback(stripped, filters=filters)
