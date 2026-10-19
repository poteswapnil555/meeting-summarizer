"""Usage: python -m meeting_summarizer TRANSCRIPT_FILE [--format auto|text|vtt] [--chunk-chars N] [--json]"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .llm import OpenAICompatibleClient
from .summarizer import MeetingSummary, summarize


def _where(item) -> str:
    return " ".join(x for x in (f"[{item.time}]" if item.time else "", f"{item.speaker}:" if item.speaker else "") if x)


def format_summary(r: MeetingSummary) -> str:
    out = ["SUMMARY", r.summary or "(none)", ""]
    out.append("DECISIONS")
    out += [f"- {d.text}\n    {_where(d)} \"{d.quote}\"" for d in r.decisions] or ["- none found"]
    out.append("\nACTION ITEMS")
    for a in r.action_items:
        out.append(f"- {a.owner or 'unassigned'}: {a.text}" + (f" (due: {a.due})" if a.due else ""))
        out.append(f"    {_where(a)} \"{a.quote}\"")
        out += [f"    note: {f}" for f in a.flags]
    if not r.action_items:
        out.append("- none found")
    out.append("\nOPEN QUESTIONS")
    out += [f"- {q.text}\n    {_where(q)} \"{q.quote}\"" for q in r.open_questions] or ["- none found"]
    if r.rejected:
        out.append(f"\nDROPPED: {len(r.rejected)} item(s) proposed by the model were not backed by the transcript")
        out += [f"- {x.kind}: {x.reason}" for x in r.rejected]
    out += [f"\nWARNING: {w}" for w in r.warnings]
    out.append(f"\n({r.chunks} chunk(s), {r.llm_calls} LLM call(s), status: {r.status})")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--format", choices=["auto", "text", "vtt"], default="auto")
    ap.add_argument("--chunk-chars", type=int)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)

    try:
        text = args.file.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"Cannot read {args.file}: {e}", file=sys.stderr)
        return 2
    try:
        llm = OpenAICompatibleClient()
    except (RuntimeError, ValueError, ImportError) as e:
        print(f"Cannot start: {e}", file=sys.stderr)
        return 2

    try:
        result = summarize(text, llm, chunk_chars=args.chunk_chars, fmt=args.format)
    except (RuntimeError, ValueError) as e:  # for example a malformed CHUNK_CHARS variable
        print(f"Cannot run: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
    else:
        print(format_summary(result))
    if result.status in {"empty_transcript", "too_long", "failed"}:
        print(f"Result: {result.status}. {result.error or ''}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
