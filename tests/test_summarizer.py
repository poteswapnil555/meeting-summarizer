import json

import pytest

from conftest import DECISION, QA_TASK, QUESTION, SAM_TASK, TEXT, RouterLLM, SequenceLLM, reply
from meeting_summarizer.summarizer import summarize

GOOD = reply("They picked the launch date.", [DECISION], [SAM_TASK, QA_TASK], [QUESTION])


# ---- normal transcript ----------------------------------------------------------------------
def test_normal_transcript():
    llm = RouterLLM(lambda u: GOOD)
    r = summarize(TEXT, llm)
    assert r.status == "ok" and r.chunks == 1 and r.llm_calls == 1 and llm.calls == 1  # one chunk: no merge call
    assert r.summary == "They picked the launch date."
    assert [d.text for d in r.decisions] == ["Launch on March 15"] and r.decisions[0].speaker == "Priya"
    sam, qa = r.action_items
    assert (sam.owner, sam.text, sam.due, sam.speaker, sam.time) == ("Sam", "Update the release notes", "Friday", "Sam", "00:01:00")
    assert (qa.owner, qa.due, qa.speaker) == ("Priya", None, "Dana Cruz")  # speaker comes from the quoted line, not from the model
    assert [q.text for q in r.open_questions] == ["Who owns the billing migration?"] and r.rejected == [] and r.warnings == []


def test_chunk_prompt_contains_the_transcript_lines():
    llm = RouterLLM(lambda u: GOOD)
    summarize(TEXT, llm)
    assert "[00:01:00] Sam: I will update the release notes by Friday." in llm.map_prompts[0]


def test_fenced_json_reply_is_accepted():
    r = summarize(TEXT, SequenceLLM("```json\n" + json.dumps(GOOD) + "\n```"))
    assert r.status == "ok" and len(r.action_items) == 2


def test_transcript_with_no_action_items_gives_none():
    r = summarize(TEXT, RouterLLM(lambda u: reply("Just updates.")))
    assert r.status == "ok" and r.action_items == [] and r.decisions == [] and r.open_questions == []


# ---- empty / invalid input ---------------------------------------------------------------------
@pytest.mark.parametrize("text", ["", "   \n\t\n", None, 42, ["Dana: hi"], "WEBVTT\n"])
def test_empty_or_non_text_transcript_never_calls_the_llm(text):
    llm = RouterLLM(lambda u: GOOD)
    r = summarize(text, llm)
    assert r.status == "empty_transcript" and r.error and llm.calls == 0


def test_transcript_over_the_character_limit_is_refused():
    llm = RouterLLM(lambda u: GOOD)
    r = summarize("Dana: hello there. " * 100, llm, max_transcript_chars=500)
    assert r.status == "too_long" and llm.calls == 0


# ---- long transcripts: many chunks, map-reduce -----------------------------------------------------
def long_text(n):
    return "\n".join(f"[{i // 60:02d}:{i % 60:02d}:00] Speaker{i % 4}: This is turn number {i} of a long meeting about topic {i}." for i in range(n))


def test_very_long_transcript_is_split_and_merged_hierarchically():
    llm = RouterLLM(lambda u: reply(f"Summary of {u.splitlines()[0]}"))
    r = summarize(long_text(600), llm, chunk_chars=1000, max_chunks=500)
    assert r.status == "ok" and r.chunks > 40 and len(llm.map_prompts) == r.chunks
    assert len(llm.reduce_prompts) >= 2 and r.summary == "Merged summary."  # more than one merge layer
    assert all(p.count("\n") <= 6 for p in llm.reduce_prompts)  # at most 6 partial summaries per merge call
    assert r.llm_calls == llm.calls


def test_every_chunk_gets_a_whole_set_of_turns():
    llm = RouterLLM(lambda u: reply())
    summarize(long_text(100), llm, chunk_chars=800, max_chunks=100)
    joined = "\n".join(p.split("\n", 1)[1] for p in llm.map_prompts)
    assert joined.count("\n") + 1 == 100 and "turn number 99 " in joined  # all 100 lines present exactly once


def test_chunk_limit_refuses_before_any_llm_call():
    llm = RouterLLM(lambda u: GOOD)
    r = summarize(long_text(600), llm, chunk_chars=1000, max_chunks=5)
    assert r.status == "too_long" and "MAX_CHUNKS" in r.error and llm.calls == 0 and r.chunks > 5


def test_single_huge_speaker_turn_is_split_but_quotes_are_checked_against_the_whole_turn():
    sentences = [f"Sentence number {i} says something about the plan." for i in range(300)]
    text = "Dana: " + " ".join(sentences) + " Dana will send the final budget by Monday."
    item = {"owner": "Dana", "task": "Send the final budget", "due": "Monday", "quote": "Dana will send the final budget by Monday"}
    llm = RouterLLM(lambda u: reply("part", actions=[item]) if "final budget" in u else reply("part"))
    r = summarize(text, llm, chunk_chars=1000, max_chunks=100)
    assert r.chunks > 10 and r.status == "ok"
    assert len(r.action_items) == 1 and r.action_items[0].owner == "Dana" and r.action_items[0].due == "Monday"


def test_duplicate_items_from_different_chunks_are_merged_once():
    llm = RouterLLM(lambda u: reply("p", actions=[SAM_TASK]))
    r = summarize(TEXT + long_text(50), llm, chunk_chars=300, max_chunks=100)
    assert r.chunks > 3 and len(r.action_items) == 1


def test_items_come_back_in_transcript_order():
    r = summarize(TEXT, RouterLLM(lambda u: reply("p", actions=[QA_TASK, SAM_TASK])))
    assert [a.owner for a in r.action_items] == ["Sam", "Priya"]  # Sam spoke first in the transcript


def test_reduce_step_sees_only_summaries_not_items():
    llm = RouterLLM(lambda u: reply(f"Summary {len(u)}", [DECISION], [SAM_TASK]) if "Sam:" in u else reply("Other"))
    summarize(TEXT, llm, chunk_chars=300)
    for p in llm.reduce_prompts:
        assert "I will update" not in p and "quote" not in p


# ---- bad JSON and retries ------------------------------------------------------------------------------------
def test_bad_json_then_valid_retries_with_the_error():
    llm = SequenceLLM("Sure! Here is the summary", GOOD)
    r = summarize(TEXT, llm)
    assert r.status == "ok" and r.llm_calls == 2 and "previous answer was invalid" in llm.prompts[1]


@pytest.mark.parametrize("bad", ["", "null", "[]", "42", "{}", '{"summary": 5}',
                                 json.dumps({"summary": "s", "decisions": [], "action_items": "none", "open_questions": []})])
def test_wrong_shape_is_retried_then_failed_cleanly(bad):
    llm = SequenceLLM(bad)
    r = summarize(TEXT, llm)
    assert r.status == "failed" and r.llm_calls == 2 and r.decisions == [] and r.action_items == []
    assert r.warnings and "could not be processed" in r.warnings[0]


def test_wrong_shape_then_valid():
    r = summarize(TEXT, SequenceLLM('{"summary": "s"}', GOOD))
    assert r.status == "ok" and len(r.action_items) == 2


def test_one_bad_chunk_gives_a_partial_result_not_a_crash():
    state = {"n": 0}

    def fn(u):
        state["n"] += 1
        return "not json" if "Part 1" in u or "part 1 of" in u.lower() else reply(f"ok {state['n']}", actions=[SAM_TASK])

    r = summarize(TEXT + long_text(40), RouterLLM(fn), chunk_chars=500, max_chunks=100)
    assert r.status == "partial" and r.chunks > 2 and len(r.action_items) == 1
    assert any("part 1/" in w for w in r.warnings)


# ---- anti-invention ---------------------------------------------------------------------------------------------
def test_action_item_with_quote_not_in_transcript_is_rejected_and_listed():
    invented = {"owner": "Sam", "task": "Ship the invoice module", "due": "Friday", "quote": "I will ship the invoice module by Friday"}
    r = summarize(TEXT, RouterLLM(lambda u: reply("s", actions=[SAM_TASK, invented])))
    assert [a.text for a in r.action_items] == ["Update the release notes"]
    assert len(r.rejected) == 1 and r.rejected[0].kind == "action_item" and r.rejected[0].reason == "quote not found in the transcript"
    assert r.rejected[0].item["task"] == "Ship the invoice module"


def test_invented_items_in_a_meeting_without_actions_never_appear():
    chat = "Dana: Nice weather today, isn't it?\nSam: Yes, really lovely. I enjoyed the walk this morning."
    invented = {"owner": "Sam", "task": "Book a venue", "due": "Friday", "quote": "Sam will book a venue by Friday"}
    d = {"text": "Move the office", "quote": "we decided to move the office"}
    r = summarize(chat, RouterLLM(lambda u: reply("Small talk.", [d], [invented], [])))
    assert r.action_items == [] and r.decisions == [] and len(r.rejected) == 2


def test_unknown_owner_is_cleared_and_flagged_but_the_backed_item_is_kept():
    task = {**SAM_TASK, "owner": "Zed Mallory"}
    r = summarize(TEXT, RouterLLM(lambda u: reply("s", actions=[task])))
    (a,) = r.action_items
    assert a.owner is None and any("Zed Mallory" in f for f in a.flags) and r.rejected == []


def test_due_date_not_in_transcript_is_cleared_and_flagged():
    task = {**SAM_TASK, "due": "2024-06-07"}
    (a,) = summarize(TEXT, RouterLLM(lambda u: reply("s", actions=[task]))).action_items
    assert a.due is None and any("due date" in f for f in a.flags)


def test_generic_owner_becomes_unassigned_without_a_flag():
    (a,) = summarize(TEXT, RouterLLM(lambda u: reply("s", actions=[{**SAM_TASK, "owner": "the team"}]))).action_items
    assert a.owner is None and a.flags == []


def test_malformed_items_are_rejected_not_crashing():
    r = summarize(TEXT, RouterLLM(lambda u: reply("s", [None, "x", {"text": 5}], [{"task": "no quote"}, 7], [{}])))
    assert r.status == "ok" and r.decisions == [] and r.action_items == [] and len(r.rejected) == 6


# ---- VTT end to end -------------------------------------------------------------------------------------------------------
def test_vtt_transcript_end_to_end():
    vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Dana>Let's wrap up.</v>\n\n"
           "00:00:05.000 --> 00:00:08.000\n<v Sam>I will update the release notes by Friday.</v>\n")
    llm = RouterLLM(lambda u: reply("s", actions=[SAM_TASK]))
    (a,) = summarize(vtt, llm).action_items
    assert a.speaker == "Sam" and a.time == "00:00:05" and "[00:00:05] Sam: I will update" in llm.map_prompts[0]


# ---- LLM failures --------------------------------------------------------------------------------------------------------------
def test_llm_timeout_is_a_failed_result_not_a_crash():
    llm = SequenceLLM(TimeoutError("timed out"))
    r = summarize(TEXT, llm)
    assert r.status == "failed" and "TimeoutError" in r.warnings[0] and r.summary == ""


def test_two_consecutive_llm_errors_stop_the_remaining_chunks():
    llm = SequenceLLM(TimeoutError("down"))
    r = summarize(long_text(100), llm, chunk_chars=500, max_chunks=100)
    assert r.status == "failed" and len(llm.prompts) == 2 and any("consecutive" in w for w in r.warnings)


def test_recovering_after_one_llm_error_continues():
    llm = SequenceLLM(TimeoutError("blip"), reply("second part"))
    r = summarize(long_text(30), llm, chunk_chars=600, max_chunks=100)
    assert r.status == "partial" and r.summary


def test_failed_merge_step_keeps_the_partial_summaries():
    class MergeFails(RouterLLM):
        def complete(self, system, user):
            if system.startswith("You merge"):
                raise TimeoutError("merge down")
            return super().complete(system, user)

    r = summarize(long_text(30), MergeFails(lambda u: reply("Part summary.")), chunk_chars=600, max_chunks=100)
    assert r.status == "ok" and r.summary.count("Part summary.") == r.chunks and any("merging" in w for w in r.warnings)


def test_invalid_merge_reply_falls_back_to_joined_summaries():
    class BadMerge(RouterLLM):
        def complete(self, system, user):
            return "garbage" if system.startswith("You merge") else super().complete(system, user)

    r = summarize(long_text(30), BadMerge(lambda u: reply("Part summary.")), chunk_chars=600, max_chunks=100)
    assert "Part summary." in r.summary and r.warnings
