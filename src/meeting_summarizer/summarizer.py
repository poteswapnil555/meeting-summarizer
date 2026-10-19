"""Map-reduce meeting summariser.

1. parse the transcript into speaker turns; split into chunks (whole turns, see chunker.py)
2. MAP: the LLM reads one chunk and returns strict JSON {summary, decisions, action_items, open_questions};
   bad JSON or a wrong shape gets one retry with the problem fed back
3. every proposed item is checked IN CODE against the full transcript (verify.py); unbacked items are rejected
4. REDUCE: only the chunk summaries (prose) go to the LLM to be merged; decisions, action items and
   questions are merged by code from the verified items, so a merge step can never add an invented item
Failures never raise: a chunk that cannot be processed is reported and the rest is still used.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .chunker import chunk_turns
from .config import env_int
from .llm import LLMClient
from .transcript import parse_transcript
from .verify import Item, Rejected, Verifier, dedupe, verify_items

REDUCE_GROUP = 6
MAX_CONSECUTIVE_LLM_ERRORS = 2

MAP_SYSTEM = """You summarise one part of a meeting transcript. Use ONLY what is said in the text. Never guess or add anything.
Return ONLY one JSON object with exactly these keys:
{"summary": "2-4 sentences about this part",
 "decisions": [{"text": "what was decided", "quote": "..."}],
 "action_items": [{"owner": "person who will do it, or null", "task": "what must be done",
                   "due": "the deadline exactly as spoken (for example Friday), or null", "quote": "..."}],
 "open_questions": [{"text": "a question left unanswered", "quote": "..."}]}
Rules:
- Every decision, action item and open question MUST have a "quote": a sentence copied word for word from ONE line of the transcript.
- Use a person's name only if it appears in the transcript. Use null for an owner or due date that is not stated.
- Do not turn ideas, suggestions or discussion into decisions or action items. Empty lists are fine and expected when nothing qualifies.
- Do not convert dates; copy the deadline as it was said."""

REDUCE_SYSTEM = """You merge partial summaries of consecutive parts of ONE meeting into a single short summary (at most 6 sentences).
Use ONLY the information in the partial summaries. Do not add facts, decisions, tasks or names that are not in them.
Return ONLY one JSON object: {"summary": "..."}"""


@dataclass
class MeetingSummary:
    status: str  # ok | partial | failed | empty_transcript | too_long
    summary: str = ""
    decisions: list[Item] = field(default_factory=list)
    action_items: list[Item] = field(default_factory=list)
    open_questions: list[Item] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)  # proposed by the model, not backed by the transcript
    warnings: list[str] = field(default_factory=list)
    chunks: int = 0
    llm_calls: int = 0
    error: str | None = None


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    data = json.loads(fenced.group(1) if fenced else raw)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    return data


def validate_chunk_reply(data: dict) -> None:
    if not isinstance(data.get("summary"), str):
        raise ValueError('"summary" must be a string')
    for key in ("decisions", "action_items", "open_questions"):
        if not isinstance(data.get(key), list):
            raise ValueError(f'"{key}" must be a list (use [] when there are none)')


def validate_reduce_reply(data: dict) -> None:
    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
        raise ValueError('"summary" must be a non-empty string')


class _Calls:
    """Calls the LLM expecting strict JSON; retries once with the problem fed back."""

    def __init__(self, llm: LLMClient, max_retries: int) -> None:
        self.llm, self.max_retries, self.count = llm, max_retries, 0

    def json_call(self, system: str, user: str, validate) -> dict:
        problem = ""
        for _ in range(self.max_retries + 1):
            prompt = user if not problem else f"{user}\n\nYour previous answer was invalid: {problem}\nReturn corrected JSON only."
            self.count += 1
            raw = self.llm.complete(system, prompt)  # exceptions (timeout, 429 exhausted) propagate to the caller
            try:
                data = parse_json(raw)
                validate(data)
                return data
            except ValueError as e:  # includes JSONDecodeError
                problem = f"{type(e).__name__}: {str(e)[:150]}"
        raise ValueError(f"the model did not return valid JSON after retries ({problem})")


def _reduce(summaries: list[str], calls: _Calls, warnings: list[str]) -> str:
    layer = [s.strip() for s in summaries if s.strip()]
    if not layer:
        return ""
    while len(layer) > 1:
        merged = []
        for i in range(0, len(layer), REDUCE_GROUP):
            group = layer[i:i + REDUCE_GROUP]
            if len(group) == 1:
                merged.append(group[0])
                continue
            user = "Partial summaries, in order:\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(group, start=1))
            try:
                merged.append(calls.json_call(REDUCE_SYSTEM, user, validate_reduce_reply)["summary"].strip())
            except Exception as e:  # keep the information rather than lose it
                warnings.append(f"merging summaries failed ({type(e).__name__}); partial summaries were joined as they are")
                merged.append(" ".join(group))
        layer = merged
    return layer[0]


def summarize(text: object, llm: LLMClient, chunk_chars: int | None = None, max_chunks: int | None = None,
              max_retries: int = 1, fmt: str = "auto", max_transcript_chars: int | None = None) -> MeetingSummary:
    chunk_chars = chunk_chars if chunk_chars is not None else env_int("CHUNK_CHARS", 6000)
    max_chunks = max_chunks if max_chunks is not None else env_int("MAX_CHUNKS", 30)
    limit = max_transcript_chars if max_transcript_chars is not None else env_int("MAX_TRANSCRIPT_CHARS", 400_000)

    if isinstance(text, str) and len(text) > limit:
        return MeetingSummary("too_long", error=f"the transcript is {len(text)} characters; the limit is {limit}")
    turns, warnings = parse_transcript(text, fmt)
    if not turns:
        return MeetingSummary("empty_transcript", warnings=warnings, error="the transcript has no readable text")
    chunks = chunk_turns(turns, chunk_chars)
    if len(chunks) > max_chunks:
        return MeetingSummary("too_long", warnings=warnings, chunks=len(chunks),
                              error=f"the transcript needs {len(chunks)} chunks; the limit is {max_chunks} (raise MAX_CHUNKS or CHUNK_CHARS)")

    verifier, calls = Verifier(turns), _Calls(llm, max_retries)
    summaries: list[str] = []
    items: dict[str, list[Item]] = {"decision": [], "action_item": [], "open_question": []}
    rejected: list[Rejected] = []
    failed = consecutive_llm_errors = 0

    for chunk in chunks:
        if consecutive_llm_errors >= MAX_CONSECUTIVE_LLM_ERRORS:
            failed += 1
            continue
        try:
            data = calls.json_call(MAP_SYSTEM, f"Transcript part {chunk.index + 1} of {len(chunks)}:\n{chunk.text}", validate_chunk_reply)
            consecutive_llm_errors = 0
        except ValueError as e:  # bad JSON after retries: this chunk is lost, the next may be fine
            failed += 1
            warnings.append(f"part {chunk.index + 1}/{len(chunks)} could not be processed: {e}")
            continue
        except Exception as e:  # timeout, connection error, exhausted rate limit
            failed += 1
            consecutive_llm_errors += 1
            warnings.append(f"part {chunk.index + 1}/{len(chunks)} failed: {type(e).__name__}: {str(e)[:150]}")
            continue
        summaries.append(data["summary"])
        for kind, key in (("decision", "decisions"), ("action_item", "action_items"), ("open_question", "open_questions")):
            ok, bad = verify_items(kind, data[key], verifier)
            items[kind] += ok
            rejected += bad
    if consecutive_llm_errors >= MAX_CONSECUTIVE_LLM_ERRORS:
        warnings.append(f"stopped after {MAX_CONSECUTIVE_LLM_ERRORS} consecutive LLM errors; the remaining parts were skipped")

    if failed == len(chunks):
        return MeetingSummary("failed", warnings=warnings, chunks=len(chunks), llm_calls=calls.count,
                              rejected=rejected, error="no part of the transcript could be summarised")
    return MeetingSummary(
        status="partial" if failed else "ok",
        summary=_reduce(summaries, calls, warnings),
        decisions=dedupe(items["decision"]),
        action_items=dedupe(items["action_item"]),
        open_questions=dedupe(items["open_question"]),
        rejected=rejected, warnings=warnings, chunks=len(chunks), llm_calls=calls.count,
    )
