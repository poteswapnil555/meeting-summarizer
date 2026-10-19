"""Anti-invention checks. Pure code: nothing the model says is kept unless the transcript backs it.

Every decision, action item and open question must carry a `quote`:
- the quote must appear (ignoring case, punctuation and spacing) inside ONE speaker turn of the transcript
  and have at least MIN_QUOTE_WORDS words; otherwise the item is rejected, not shown as a result
- the speaker and time of an accepted item are taken from the matching turn by code, not from the model
Action items are checked further:
- owner: must be a speaker (full name or any part of it) or a name that appears in the transcript;
  generic owners ("we", "team", "everyone", "TBD"...) mean "no owner"; an owner not found is cleared and flagged
- due: kept exactly as the model wrote it only if that text appears in the transcript; otherwise cleared and flagged
The checks cannot tell whether a real quote supports the claim attached to it; that is a documented limitation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .transcript import Turn, fmt_time

MIN_QUOTE_WORDS = 4
GENERIC_OWNERS = frozenset({
    "we", "i", "you", "they", "he", "she", "it", "us", "me", "all", "team", "everyone", "everybody", "someone",
    "somebody", "anyone", "tbd", "tba", "unknown", "unassigned", "none", "n a", "na", "nobody", "group", "us all",
})
NO_DUE = frozenset({"", "none", "n a", "na", "null", "not stated", "unspecified", "tbd", "no due date", "unknown"})


def norm(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class Item:
    kind: str  # decision | action_item | open_question
    text: str
    quote: str
    speaker: str | None
    time: str
    turn_index: int
    owner: str | None = None  # action items only
    due: str | None = None  # action items only
    flags: list[str] = field(default_factory=list)


@dataclass
class Rejected:
    kind: str
    item: object
    reason: str


class Verifier:
    def __init__(self, turns: list[Turn]) -> None:
        self.turns = turns
        self._norms = [f" {norm(t.text)} " for t in turns]
        self._full = " ".join(self._norms)
        self._speakers: set[str] = set()
        for t in turns:
            if t.speaker:
                n = norm(t.speaker)
                self._speakers.add(n)
                self._speakers.update(w for w in n.split() if len(w) >= 2)

    def find_quote(self, quote: object) -> tuple[Turn | None, str]:
        """Return (turn containing the quote, reason if not found)."""
        if not isinstance(quote, str) or not quote.strip():
            return None, "no quote given"
        q = norm(quote)
        if len(q.split()) < MIN_QUOTE_WORDS:
            return None, f"quote too short (needs at least {MIN_QUOTE_WORDS} words)"
        needle = f" {q} "
        for turn, n in zip(self.turns, self._norms):
            if needle in n:
                return turn, ""
        return None, "quote not found in the transcript"

    def check_owner(self, owner: object) -> tuple[str | None, str | None]:
        """Return (owner or None, flag or None)."""
        if owner is None:
            return None, None
        if not isinstance(owner, str):
            return None, "owner is not text; cleared"
        n = re.sub(r"^(?:the|a|an)\s+", "", norm(owner))  # "the team" is as generic as "team"
        if n in GENERIC_OWNERS or len(n) < 2:
            return None, None
        if n in self._speakers or f" {n} " in self._full:
            return owner.strip(), None
        return None, f"owner '{owner.strip()[:40]}' not found in the transcript; owner cleared"

    def check_due(self, due: object) -> tuple[str | None, str | None]:
        if due is None:
            return None, None
        if not isinstance(due, str):
            return None, "due date is not text; cleared"
        n = norm(due)
        if n in NO_DUE:
            return None, None
        if f" {n} " in self._full:
            return due.strip(), None
        return None, f"due date '{due.strip()[:40]}' not found in the transcript; cleared"


def verify_items(kind: str, raw_items: object, verifier: Verifier) -> tuple[list[Item], list[Rejected]]:
    """Check the items the model proposed for one kind. Returns (accepted, rejected)."""
    accepted: list[Item] = []
    rejected: list[Rejected] = []
    if not isinstance(raw_items, list):
        return accepted, rejected
    for raw in raw_items:
        if not isinstance(raw, dict):
            rejected.append(Rejected(kind, raw, "item is not an object"))
            continue
        main_key = "task" if kind == "action_item" else "text"
        text = raw.get(main_key)
        if not isinstance(text, str) or not text.strip():
            rejected.append(Rejected(kind, raw, f"missing '{main_key}'"))
            continue
        turn, reason = verifier.find_quote(raw.get("quote"))
        if turn is None:
            rejected.append(Rejected(kind, raw, reason))
            continue
        item = Item(kind, text.strip(), str(raw["quote"]).strip(), turn.speaker, fmt_time(turn.start), turn.index)
        if kind == "action_item":
            item.owner, flag = verifier.check_owner(raw.get("owner"))
            item.flags += [flag] if flag else []
            item.due, flag = verifier.check_due(raw.get("due"))
            item.flags += [flag] if flag else []
        accepted.append(item)
    return accepted, rejected


def dedupe(items: list[Item]) -> list[Item]:
    """Drop repeats (same quote, or same owner and task) and keep transcript order."""
    seen_quotes: set[tuple[str, str]] = set()
    seen_tasks: set[tuple[str, str, str]] = set()
    out: list[Item] = []
    for item in sorted(items, key=lambda i: i.turn_index):
        qkey = (item.kind, norm(item.quote))
        tkey = (item.kind, norm(item.owner or ""), norm(item.text))
        if qkey in seen_quotes or tkey in seen_tasks:
            continue
        seen_quotes.add(qkey)
        seen_tasks.add(tkey)
        out.append(item)
    return out
