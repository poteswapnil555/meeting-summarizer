import json

import pytest

from meeting_summarizer.summarizer import REDUCE_SYSTEM

TEXT = """[00:00:05] Dana Cruz: Let's start. The goal today is to pick the launch date.
[00:00:40] Priya: Okay, we decided to launch on March 15.
[00:01:00] Sam: I will update the release notes by Friday.
[00:01:20] Dana Cruz: Priya, please confirm the QA schedule with the vendor.
[00:01:40] Priya: Sure. Does anyone know who owns the billing migration?
"""

DECISION = {"text": "Launch on March 15", "quote": "we decided to launch on March 15"}
SAM_TASK = {"owner": "Sam", "task": "Update the release notes", "due": "Friday", "quote": "I will update the release notes by Friday"}
QA_TASK = {"owner": "Priya", "task": "Confirm the QA schedule with the vendor", "due": None,
           "quote": "Priya, please confirm the QA schedule with the vendor"}
QUESTION = {"text": "Who owns the billing migration?", "quote": "Does anyone know who owns the billing migration"}


def reply(summary="A summary.", decisions=(), actions=(), questions=()):
    return {"summary": summary, "decisions": list(decisions), "action_items": list(actions), "open_questions": list(questions)}


class RouterLLM:
    """Fake LLM. map_fn(user_prompt) -> dict (or a raw string) for chunk calls; reduce replies with a fixed summary."""

    def __init__(self, map_fn, reduce_text="Merged summary."):
        self.map_fn, self.reduce_text = map_fn, reduce_text
        self.map_prompts: list[str] = []
        self.reduce_prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        if system == REDUCE_SYSTEM:
            self.reduce_prompts.append(user)
            return json.dumps({"summary": self.reduce_text})
        self.map_prompts.append(user)
        out = self.map_fn(user)
        return out if isinstance(out, str) else json.dumps(out)

    @property
    def calls(self) -> int:
        return len(self.map_prompts) + len(self.reduce_prompts)


class SequenceLLM:
    """Returns queued raw replies in order (the last repeats); records prompts. Exceptions in the queue are raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


@pytest.fixture
def text():
    return TEXT
