"""Fail-closed citation validation against the turn registry."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from app.assistant.deps import TurnRegistry
from app.assistant.outputs import GroundedAnswer

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    error: str | None = None


class CitationGroundingCase(BaseModel):
    citation_index: int
    answer: str
    excerpt: str
    source_text: str


class CitationGroundingDecision(BaseModel):
    citation_index: int
    supported: bool
    reason: str = Field(description="Short reason for the grounding decision")


class CitationGroundingDecisionList(BaseModel):
    decisions: list[CitationGroundingDecision]


class GroundingJudge(Protocol):
    async def judge(
        self,
        cases: list[CitationGroundingCase],
    ) -> list[CitationGroundingDecision]: ...


class LocalGroundingJudge:
    async def judge(
        self,
        cases: list[CitationGroundingCase],
    ) -> list[CitationGroundingDecision]:
        return await asyncio.to_thread(self._judge_sync, cases)

    def _judge_sync(
        self,
        cases: list[CitationGroundingCase],
    ) -> list[CitationGroundingDecision]:
        return [_judge_case(case) for case in cases]


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _content_words(text: str) -> set[str]:
    return {
        word.casefold()
        for word in _WORD_RE.findall(text)
        if len(word) > 2 and word.casefold() not in _STOP_WORDS
    }


def _judge_case(case: CitationGroundingCase) -> CitationGroundingDecision:
    source = _normalize_text(case.source_text)
    excerpt = _normalize_text(case.excerpt)

    if excerpt and excerpt in source:
        return CitationGroundingDecision(
            citation_index=case.citation_index,
            supported=True,
            reason="citation excerpt appears in the retrieved source text",
        )

    excerpt_words = _content_words(case.excerpt)
    source_words = _content_words(case.source_text)
    if excerpt_words and excerpt_words.issubset(source_words):
        return CitationGroundingDecision(
            citation_index=case.citation_index,
            supported=True,
            reason="citation excerpt terms appear in the retrieved source text",
        )

    return CitationGroundingDecision(
        citation_index=case.citation_index,
        supported=False,
        reason="citation excerpt was not found in the retrieved source text",
    )


def _citation_markers(text: str) -> set[int]:
    return {int(match.group(1)) for match in _CITATION_MARKER_RE.finditer(text)}


def prune_unreferenced_citations(answer: GroundedAnswer) -> GroundedAnswer:
    marker_indices = _citation_markers(answer.answer)
    if not marker_indices:
        return answer

    citations = [
        citation
        for citation in answer.citations
        if citation.citation_index in marker_indices
    ]
    if len(citations) == len(answer.citations):
        return answer

    return answer.model_copy(update={"citations": citations})


def _decision_indexes_match_cases(
    decisions: list[CitationGroundingDecision],
    cases: list[CitationGroundingCase],
) -> bool:
    decision_indices = [decision.citation_index for decision in decisions]
    if len(decision_indices) != len(set(decision_indices)):
        return False
    return set(decision_indices) == {case.citation_index for case in cases}


async def _judge_with_index_repair(
    judge: GroundingJudge,
    cases: list[CitationGroundingCase],
) -> list[CitationGroundingDecision]:
    decisions = await judge.judge(cases)
    if _decision_indexes_match_cases(decisions, cases) or len(cases) <= 1:
        return decisions

    repaired: list[CitationGroundingDecision] = []
    for case in cases:
        repaired.extend(await judge.judge([case]))
    return repaired


class GroundingValidator:
    def __init__(self, judge: GroundingJudge | None = None) -> None:
        self._judge = judge

    async def validate(
        self,
        answer: GroundedAnswer,
        registry: TurnRegistry,
    ) -> ValidationResult:
        if not answer.answer.strip():
            return ValidationResult(ok=False, error="Answer text is empty.")

        if answer.insufficient_evidence:
            if answer.citations:
                return ValidationResult(
                    ok=False,
                    error="insufficient_evidence answers must not include citations.",
                )
            return ValidationResult(ok=True)

        if not answer.citations:
            return ValidationResult(
                ok=False,
                error="Grounded answers must include at least one citation.",
            )

        if not registry.passages_by_chunk_id:
            return ValidationResult(
                ok=False,
                error="Citations present but no passages were retrieved this turn.",
            )

        indices = [citation.citation_index for citation in answer.citations]
        if len(indices) != len(set(indices)):
            return ValidationResult(ok=False, error="Duplicate citation_index values.")

        expected_indices = list(range(1, len(indices) + 1))
        if sorted(indices) != expected_indices:
            return ValidationResult(
                ok=False,
                error="citation_index values must be unique, 1-based, and contiguous.",
            )

        marker_indices = _citation_markers(answer.answer)
        if marker_indices != set(indices):
            return ValidationResult(
                ok=False,
                error="Answer [n] markers must match citation_index values exactly.",
            )

        cases: list[CitationGroundingCase] = []
        for citation in answer.citations:
            passage = registry.passages_by_chunk_id.get(citation.chunk_id)
            if passage is None:
                return ValidationResult(
                    ok=False,
                    error=f"Citation references chunk {citation.chunk_id} that was not retrieved.",
                )
            cases.append(
                CitationGroundingCase(
                    citation_index=citation.citation_index,
                    answer=answer.answer,
                    excerpt=citation.excerpt,
                    source_text=passage.text,
                )
            )

        try:
            judge = self._judge or LocalGroundingJudge()
            decisions = await _judge_with_index_repair(judge, cases)
        except Exception as exc:
            return ValidationResult(
                ok=False,
                error=f"Grounding judge failed: {exc}",
            )

        decision_indices = [decision.citation_index for decision in decisions]
        if len(decision_indices) != len(set(decision_indices)):
            return ValidationResult(
                ok=False,
                error="Grounding judge returned duplicate citation decisions.",
            )

        decision_by_index = {
            decision.citation_index: decision for decision in decisions
        }
        if set(decision_by_index) != set(indices):
            return ValidationResult(
                ok=False,
                error="Grounding judge decisions must match citation indexes exactly.",
            )

        for citation_index in indices:
            decision = decision_by_index[citation_index]
            if not decision.supported:
                return ValidationResult(
                    ok=False,
                    error=(
                        f"Citation [{citation_index}] is not supported by retrieved "
                        f"source text: {decision.reason}"
                    ),
                )

        return ValidationResult(ok=True)
