"""Coordinates one chat turn: agent → validate → stream → persist."""

from __future__ import annotations

import asyncio
import traceback
import uuid
from collections.abc import AsyncIterator

from supabase import AsyncClient

from app.assistant.agent import run_document_agent
from app.assistant.deps import DocumentAgentDeps, TurnRegistry
from app.assistant.outputs import Citation, GroundedAnswer
from app.auth.dependencies import CurrentUser
from app.chat.messages import text_from_parts
from app.chat.streaming import (
    stream_grounded_turn_and_persist,
    stream_error,
    stream_status,
)
from app.grounding.validator import (
    GroundingValidator,
    prune_unreferenced_citations,
)
from app.retrieval.retriever import DocumentRetriever
from app.schemas.chat import UIMessage

MAX_VALIDATION_ATTEMPTS = 2
FALLBACK_CITATION_COUNT = 3
FALLBACK_EXCERPT_CHARS = 280


async def _yield_status_updates(
    status_queue: asyncio.Queue[tuple[str, str]],
    agent_task: asyncio.Task[GroundedAnswer],
) -> AsyncIterator[str]:
    while not agent_task.done():
        try:
            stage, message = await asyncio.wait_for(
                status_queue.get(),
                timeout=0.3,
            )
        except TimeoutError:
            continue

        async for event in stream_status(stage, message):
            yield event

    while not status_queue.empty():
        stage, message = status_queue.get_nowait()
        async for event in stream_status(stage, message):
            yield event


def _fallback_grounded_answer(
    query: str,
    *,
    registry: TurnRegistry,
    retriever: DocumentRetriever,
) -> tuple[GroundedAnswer, TurnRegistry]:
    passages = list(registry.passages_by_chunk_id.values())
    if not passages:
        passages = retriever.search(query, top_k=FALLBACK_CITATION_COUNT)
        registry.register_many(passages)

    passages = sorted(
        passages,
        key=lambda passage: passage.fusion_score,
        reverse=True,
    )[:FALLBACK_CITATION_COUNT]

    if not passages:
        return (
            GroundedAnswer(
                answer=(
                    "I could not find enough evidence in the filing corpus to answer "
                    "that question."
                ),
                citations=[],
                insufficient_evidence=True,
            ),
            registry,
        )

    citations: list[Citation] = []
    lines = [
        "I found relevant filing passages, but the local Ollama model could not "
        "complete the full analysis. Here are the strongest retrieved sources:"
    ]

    for index, passage in enumerate(passages, start=1):
        excerpt = " ".join(passage.text.strip().split())
        if len(excerpt) > FALLBACK_EXCERPT_CHARS:
            excerpt = excerpt[:FALLBACK_EXCERPT_CHARS].rstrip() + "..."

        year = passage.fiscal_year or passage.filing_date.year
        section = f", {passage.section}" if passage.section else ""
        page = f", page {passage.page}" if passage.page else ""
        lines.append(f"{index}. {passage.ticker} {passage.form} FY{year}{section}{page}: {excerpt} [{index}]")
        citations.append(
            Citation(
                citation_index=index,
                chunk_id=passage.chunk_id,
                excerpt=excerpt,
            )
        )

    return (
        GroundedAnswer(
            answer="\n".join(lines),
            citations=citations,
        ),
        registry,
    )


async def run_turn(
    *,
    client: AsyncClient,
    thread_id: uuid.UUID,
    user: CurrentUser,
    user_message: UIMessage,
    thread_title: str,
    retriever: DocumentRetriever,
) -> AsyncIterator[str]:

    print("\n" + "=" * 80)
    print("run_turn() started")
    print("=" * 80)

    loop = asyncio.get_running_loop()
    query = text_from_parts(user_message.parts).strip()

    print(f"Query: {query}")

    if not query:
        async for event in stream_error("User message is empty."):
            yield event
        return

    async for event in stream_status(
        "analyzing",
        "Analyzing your question…",
    ):
        yield event

    grounded: GroundedAnswer | None = None
    validation = None

    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        print(f"\nAttempt {attempt}")

        registry = TurnRegistry()
        status_queue = asyncio.Queue()

        def on_status(stage: str, message: str) -> None:
            print(f"[STATUS] {stage}: {message}")
            loop.call_soon_threadsafe(
                status_queue.put_nowait,
                (stage, message),
            )

        deps = DocumentAgentDeps(
            retriever=retriever,
            registry=registry,
            thread_id=thread_id,
            user_id=user.id,
            on_status=on_status,
        )

        print("Launching run_document_agent()...")

        agent_task = asyncio.create_task(
            asyncio.to_thread(
                run_document_agent,
                query,
                deps,
            )
        )

        async for event in _yield_status_updates(
            status_queue,
            agent_task,
        ):
            yield event

        try:
            grounded = await agent_task
            print("Agent completed successfully.")

        except Exception as exc:
            print("\n" + "=" * 80)
            print("AGENT EXCEPTION")
            traceback.print_exc()
            print("=" * 80)

            async for event in stream_status(
                "retrying",
                "Local model output was invalid; returning retrieved sources instead.",
            ):
                yield event

            grounded, registry = _fallback_grounded_answer(
                query,
                registry=registry,
                retriever=retriever,
            )
            validation = await GroundingValidator().validate(grounded, registry)
            break

        async for event in stream_status(
            "verifying",
            "Verifying citations…",
        ):
            yield event

        grounded = prune_unreferenced_citations(
            grounded,
        )

        validation = await GroundingValidator().validate(
            grounded,
            registry,
        )

        if validation.ok or attempt == MAX_VALIDATION_ATTEMPTS:
            break

        async for event in stream_status(
            "retrying",
            "Could not fully verify citations; retrying with stricter grounding…",
        ):
            yield event

    if grounded is None or validation is None:
        async for event in stream_error(
            "Assistant run failed before producing an answer."
        ):
            yield event
        return

    if validation.ok:
        async for event in stream_status(
            "streaming",
            "Preparing answer…",
        ):
            yield event

    async for event in stream_grounded_turn_and_persist(
        client=client,
        thread_id=thread_id,
        user=user,
        user_message=user_message,
        thread_title=thread_title,
        answer=grounded,
        registry=registry,
        validation=validation,
    ):
        yield event
