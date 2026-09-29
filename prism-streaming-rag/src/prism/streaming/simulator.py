"""
PRISM Streaming Transcript Simulator

Provides:
  - TranscriptChunk: a single emitted chunk with metadata
  - TranscriptSimulator: deterministic chunk-by-chunk emitter
  - StreamingTranscriptSession: live session tracking across chunks

This is the primary input interface for the Adaptive Controller.
No microphone hardware required for benchmark/evaluation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional


# ──────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────

@dataclass
class TranscriptChunk:
    """A single unit of streaming transcript."""

    chunk_index: int
    text: str                        # Cumulative transcript up to this chunk
    delta_text: str                  # New text added in this chunk
    session_id: str
    timestamp: float = field(default_factory=time.time)
    is_final: bool = False           # True for the last chunk

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def delta_word_count(self) -> int:
        return len(self.delta_text.split())

    def to_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "text": self.text,
            "delta_text": self.delta_text,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "is_final": self.is_final,
            "word_count": self.word_count,
            "delta_word_count": self.delta_word_count,
        }


# ──────────────────────────────────────────────────────────────
# Transcript Simulator
# ──────────────────────────────────────────────────────────────

class TranscriptSimulator:
    """
    Deterministic transcript simulator.

    Accepts a list of text fragments (cumulative or incremental)
    and emits them as TranscriptChunk events at configurable intervals.

    Example:
        chunks = [
            "I need to plan",
            "I need to plan a customer workshop",
            "I need to plan a customer workshop for 30 people",
            "I also need cancellation policy and catering",
        ]
        sim = TranscriptSimulator(chunks, interval_s=1.0, cumulative=True)
        async for chunk in sim.stream():
            process(chunk)

    Args:
        fragments: List of text strings. If cumulative=True, each is the
                   full transcript so far. If cumulative=False, each is
                   an incremental addition.
        interval_s: Seconds to wait between emitting chunks.
        cumulative: If True, fragments are cumulative (each replaces the
                    previous). If False, fragments are appended.
        session_id: Optional session ID. Auto-generated if not provided.
    """

    def __init__(
        self,
        fragments: List[str],
        interval_s: float = 1.0,
        cumulative: bool = True,
        session_id: Optional[str] = None,
    ) -> None:
        self.fragments = fragments
        self.interval_s = interval_s
        self.cumulative = cumulative
        self.session_id = session_id or str(uuid.uuid4())

    async def stream(self) -> AsyncIterator[TranscriptChunk]:
        """Async generator: yields TranscriptChunk with configurable delay."""
        accumulated = ""

        for idx, fragment in enumerate(self.fragments):
            if self.cumulative:
                current_text = fragment
                delta = fragment[len(accumulated):]
            else:
                delta = fragment
                current_text = accumulated + (" " if accumulated else "") + fragment

            accumulated = current_text
            is_final = idx == len(self.fragments) - 1

            chunk = TranscriptChunk(
                chunk_index=idx,
                text=current_text,
                delta_text=delta,
                session_id=self.session_id,
                is_final=is_final,
            )

            yield chunk

            if not is_final:
                await asyncio.sleep(self.interval_s)

    def stream_sync(self):
        """Synchronous generator for non-async contexts."""
        accumulated = ""

        for idx, fragment in enumerate(self.fragments):
            if self.cumulative:
                current_text = fragment
                delta = fragment[len(accumulated):]
            else:
                delta = fragment
                current_text = accumulated + (" " if accumulated else "") + fragment

            accumulated = current_text
            is_final = idx == len(self.fragments) - 1

            chunk = TranscriptChunk(
                chunk_index=idx,
                text=current_text,
                delta_text=delta,
                session_id=self.session_id,
                is_final=is_final,
            )

            yield chunk

            if not is_final:
                time.sleep(self.interval_s)


# ──────────────────────────────────────────────────────────────
# Built-in Demo Scenarios
# ──────────────────────────────────────────────────────────────

DEMO_SCENARIOS = {
    "workshop_planning": [
        "I need to plan",
        "I need to plan a customer workshop",
        "I need to plan a customer workshop for 30 people",
        "I need to plan a customer workshop for 30 people, and I also need cancellation policy and catering options",
    ],
    "reimbursement_international": [
        "Summarize the reimbursement policy",
        "Summarize the reimbursement policy for travel",
        "Summarize the reimbursement policy for travel — the trip was international",
        "Summarize the reimbursement policy for travel — the trip was international and the booking happened after travel",
    ],
    "crypto_ftx": [
        "Who is Sam",
        "Who is Sam Bankman-Fried",
        "Who is Sam Bankman-Fried and what is the FTX scandal",
        "Who is Sam Bankman-Fried, what is the FTX scandal, and what did the prosecution allege",
    ],
    "simple_suppress": [
        "What is the capital of France?",
        "Please repeat your previous answer in bullet points.",
    ],
}


def get_demo_scenario(name: str, interval_s: float = 1.0) -> TranscriptSimulator:
    """Get a named demo scenario simulator."""
    if name not in DEMO_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{name}'. Available: {list(DEMO_SCENARIOS.keys())}"
        )
    return TranscriptSimulator(
        fragments=DEMO_SCENARIOS[name],
        interval_s=interval_s,
        cumulative=True,
    )
