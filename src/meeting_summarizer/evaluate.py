"""Evaluate on hand-written synthetic transcripts with known decisions, action items and open questions.

  python -m meeting_summarizer.evaluate --mock   # plumbing check: scripted fake LLM. NOT accuracy.
  python -m meeting_summarizer.evaluate --live   # the real LLM from environment variables

A gold item counts as found when the output has an item whose text contains all of the gold keywords
("a|b" means either); an action item must also have the gold owner and, when the gold has a due date, a due date that contains it.
Items that are not in the gold list are reported as "extra" (they may be valid, but nobody wrote them down in advance).
One transcript (t4) has NO decisions or action items: the tool must return none.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .summarizer import REDUCE_SYSTEM, MeetingSummary, summarize
from .verify import Item, norm

GOLD = Path("eval/gold.json")
KINDS = (("decisions", "decision"), ("action_items", "action_item"), ("open_questions", "open_question"))


def load_gold(path: Path) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read gold file {path}: {e}") from e
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    for g in data:
        if not isinstance(g, dict) or not g.get("id") or not g.get("file"):
            raise ValueError(f"{path}: every entry needs id and file: {g}")
        for key, _ in KINDS:
            if not isinstance(g.get(key, []), list):
                raise ValueError(f"{path}: {g['id']}.{key} must be a list")
        for item in g.get("decisions", []) + g.get("action_items", []) + g.get("open_questions", []):
            if not item.get("keywords") or not item.get("quote"):
                raise ValueError(f"{path}: {g['id']}: every gold item needs keywords and a quote: {item}")
    return data


def keywords_present(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return all(any(alt.strip() in low for alt in kw.lower().split("|")) for kw in keywords)


def matches(gold: dict, item: Item, kind: str) -> bool:
    if not keywords_present(item.text, gold["keywords"]):
        return False
    if kind == "action_item":
        if norm(gold["owner"]) not in norm(item.owner or ""):
            return False
        if gold.get("due") and norm(gold["due"]) not in norm(item.due or ""):
            return False
    return True


@dataclass
class Report:
    mode: str
    counts: dict = field(default_factory=lambda: {k: {"gold": 0, "found": 0, "extra": 0} for k, _ in KINDS})
    transcripts: int = 0
    transcripts_ok: int = 0
    no_item_transcripts: int = 0
    no_item_transcripts_clean: int = 0
    rejected_by_code: int = 0
    items_with_cleared_owner_or_due: int = 0
    llm_calls: int = 0
    details: list[dict] = field(default_factory=list)

    def add(self, gold: dict, result: MeetingSummary) -> None:
        self.transcripts += 1
        self.transcripts_ok += result.status == "ok"
        self.rejected_by_code += len(result.rejected)
        self.llm_calls += result.llm_calls
        detail = {"id": gold["id"], "status": result.status, "chunks": result.chunks, "llm_calls": result.llm_calls,
                  "summary": result.summary, "warnings": result.warnings, "error": result.error,
                  "rejected": [{"kind": r.kind, "reason": r.reason, "item": r.item} for r in result.rejected], "kinds": {}}
        for key, kind in KINDS:
            produced: list[Item] = getattr(result, key)
            gold_items = gold.get(key, [])
            used: set[int] = set()
            found = []
            for g in gold_items:
                hit = next((i for i, it in enumerate(produced) if i not in used and matches(g, it, kind)), None)
                if hit is not None:
                    used.add(hit)
                found.append(hit is not None)
            extras = [it for i, it in enumerate(produced) if i not in used]
            self.counts[key]["gold"] += len(gold_items)
            self.counts[key]["found"] += sum(found)
            self.counts[key]["extra"] += len(extras)
            self.items_with_cleared_owner_or_due += sum(bool(it.flags) for it in produced)
            detail["kinds"][key] = {
                "gold_found": [{"keywords": g["keywords"], "found": f} for g, f in zip(gold_items, found)],
                "extra": [{"text": it.text, "owner": it.owner, "due": it.due, "quote": it.quote, "flags": it.flags} for it in extras],
            }
        if gold.get("expect_no_items"):
            self.no_item_transcripts += 1
            clean = not result.decisions and not result.action_items
            self.no_item_transcripts_clean += clean
            detail["expected_no_items_and_got_none"] = clean
        self.details.append(detail)

    def summary(self) -> dict:
        out = {"mode": self.mode, "transcripts": self.transcripts, "transcripts_status_ok": self.transcripts_ok}
        for key, _ in KINDS:
            c = self.counts[key]
            out[key] = {"gold": c["gold"], "found": c["found"], "extra_not_in_gold": c["extra"]}
        out.update({
            "no_item_transcripts_with_zero_items": f"{self.no_item_transcripts_clean}/{self.no_item_transcripts}",
            "proposed_items_rejected_by_code": self.rejected_by_code,
            "kept_items_with_owner_or_due_cleared": self.items_with_cleared_owner_or_due,
            "llm_calls": self.llm_calls,
        })
        return out


class MockLLM:
    """Scripted stand-in for the LLM. Plumbing only. It returns the gold items whose quote is in the chunk it is given,
    plus (for the no-item transcript) an INVENTED decision and action item that the code must reject."""

    current: dict | None = None

    def complete(self, system: str, user: str) -> str:
        g = self.current or {}
        if system == REDUCE_SYSTEM:
            return json.dumps({"summary": "Mock merged summary."})
        text = norm(user)

        def present(items):
            return [i for i in items if norm(i["quote"]) in text]

        if g.get("expect_no_items"):
            return json.dumps({"summary": "Mock status round.",
                               "decisions": [{"text": "Move the office", "quote": "we decided to move the office to Berlin"}],
                               "action_items": [{"owner": "Kim", "task": "Book a venue", "due": "Friday", "quote": "Kim will book a venue by Friday"}],
                               "open_questions": []})
        return json.dumps({
            "summary": "Mock summary.",
            "decisions": [{"text": i["text"], "quote": i["quote"]} for i in present(g.get("decisions", []))],
            "action_items": [{"owner": i["owner"], "task": i["task"], "due": i.get("due"), "quote": i["quote"]} for i in present(g.get("action_items", []))],
            "open_questions": [{"text": i["text"], "quote": i["quote"]} for i in present(g.get("open_questions", []))],
        })


def evaluate(llm, gold: list[dict], base_dir: Path, chunk_chars: int | None = None, delay: float = 0.0,
             sleep=time.sleep, progress=None, hook=None) -> Report:
    report = Report(mode="")
    for i, g in enumerate(gold, start=1):
        if hook:
            hook(g)
        text = (Path(base_dir) / g["file"]).read_text(encoding="utf-8")
        result = summarize(text, llm, chunk_chars=g.get("chunk_chars") or chunk_chars)
        report.add(g, result)
        if progress:
            progress(i, len(gold))
        if delay > 0 and i < len(gold):
            sleep(delay)
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="no key; NOT an accuracy measurement")
    group.add_argument("--live", action="store_true", help="use the LLM configured by environment variables")
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--chunk-chars", type=int, help="default: CHUNK_CHARS or 6000 (t5 always uses 1500 to force several chunks)")
    ap.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                    help="pause between transcripts, e.g. 13 for a 5 requests/minute limit")
    ap.add_argument("--out", type=Path, help="save the summary plus per-transcript details as JSON")
    args = ap.parse_args(argv)

    try:
        if args.mock:
            mode, llm = "mock", MockLLM()
            print("MODE: mock -- plumbing check with a scripted fake LLM. These numbers are NOT accuracy.")
        else:
            from .llm import OpenAICompatibleClient

            llm, mode = OpenAICompatibleClient(), "live"
            print("MODE: live -- the real LLM from environment variables.")
        gold = load_gold(args.gold)
        for g in gold:
            if not (args.gold.parent / g["file"]).is_file():
                raise ValueError(f"transcript file not found: {args.gold.parent / g['file']}")
    except (RuntimeError, ValueError, ImportError, OSError) as e:
        print(f"Cannot run evaluation: {e}", file=sys.stderr)
        return 2

    def show(done: int, total: int) -> None:
        print(f"{done}/{total} done", file=sys.stderr, flush=True)

    hook = (lambda g: setattr(llm, "current", g)) if isinstance(llm, MockLLM) else None
    report = evaluate(llm, gold, args.gold.parent, args.chunk_chars, args.delay, progress=show, hook=hook)
    report.mode = mode
    summary = report.summary()
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({**summary, "details": report.details}, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
