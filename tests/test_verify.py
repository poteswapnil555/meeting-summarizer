import pytest

from meeting_summarizer.transcript import parse_transcript
from meeting_summarizer.verify import MIN_QUOTE_WORDS, Verifier, dedupe, norm, verify_items

TEXT = """[00:00:05] Dana Cruz: Let's start. The goal today is to pick the launch date.
[00:00:40] Priya: Okay, we decided to launch on March 15.
[00:01:00] Sam: I will update the release notes by Friday.
[00:01:20] Dana Cruz: Priya, please confirm the QA schedule with the vendor.
[00:01:40] Priya: Sure. Does anyone know who owns the billing migration?
"""


@pytest.fixture
def v():
    return Verifier(parse_transcript(TEXT)[0])


def action(**kw):
    base = {"owner": "Sam", "task": "Update the release notes", "due": "Friday", "quote": "I will update the release notes by Friday"}
    return {**base, **kw}


# ---- quotes -----------------------------------------------------------------------------
def test_exact_quote_is_found_with_speaker_and_time(v):
    (item,), rejected = verify_items("action_item", [action()], v)
    assert not rejected and item.speaker == "Sam" and item.time == "00:01:00" and item.turn_index == 2


@pytest.mark.parametrize("quote", [
    "i will update the release notes by friday",
    "I WILL UPDATE THE RELEASE NOTES BY FRIDAY.",
    "I will  update the release   notes by Friday!",
    "will update the release notes",
    "“I will update the release notes”",
])
def test_quote_matching_ignores_case_punctuation_and_spacing(v, quote):
    accepted, rejected = verify_items("action_item", [action(quote=quote)], v)
    assert len(accepted) == 1 and not rejected


def test_quote_not_in_transcript_is_rejected_with_reason(v):
    accepted, rejected = verify_items("action_item", [action(quote="I will ship the invoice module by Friday")], v)
    assert accepted == [] and rejected[0].reason == "quote not found in the transcript" and rejected[0].kind == "action_item"


def test_paraphrased_quote_is_rejected(v):
    accepted, rejected = verify_items("action_item", [action(quote="Sam promised to refresh the notes before Friday")], v)
    assert accepted == [] and len(rejected) == 1


def test_quote_must_sit_inside_one_turn(v):
    spanning = "by Friday Priya please confirm the QA schedule"  # end of Sam's turn + start of Dana's
    accepted, rejected = verify_items("action_item", [action(quote=spanning)], v)
    assert accepted == [] and "not found" in rejected[0].reason


def test_quote_must_match_whole_words(v):
    accepted, _ = verify_items("decision", [{"text": "x", "quote": "we decided to launch on March 1"}], v)
    assert accepted == []  # "March 1" is only a prefix of "March 15"


@pytest.mark.parametrize("quote,reason", [
    (None, "no quote"), ("", "no quote"), ("   ", "no quote"), (42, "no quote"),
    ("update the notes", "too short"), ("a b c", "too short"),
])
def test_missing_or_short_quote_rejected(v, quote, reason):
    accepted, rejected = verify_items("action_item", [action(quote=quote)], v)
    assert accepted == [] and reason in rejected[0].reason


def test_min_quote_words_is_four():
    assert MIN_QUOTE_WORDS == 4


# ---- owners -----------------------------------------------------------------------------
@pytest.mark.parametrize("owner,expected", [
    ("Sam", "Sam"), ("sam", "sam"), ("Dana Cruz", "Dana Cruz"), ("Dana", "Dana"), ("Cruz", "Cruz"), (" Priya ", "Priya"),
])
def test_known_owners_are_kept(v, owner, expected):
    (item,), _ = verify_items("action_item", [action(owner=owner)], v)
    assert item.owner == expected and item.flags == []


def test_unknown_owner_is_cleared_and_flagged(v):
    (item,), rejected = verify_items("action_item", [action(owner="Zed Mallory")], v)
    assert item.owner is None and not rejected
    assert any("Zed Mallory" in f and "not found" in f for f in item.flags)


def test_owner_made_of_parts_of_two_different_speakers_is_not_accepted(v):
    (item,), _ = verify_items("action_item", [action(owner="Sam Cruz")], v)  # "Sam" + "Cruz" from Dana Cruz
    assert item.owner is None and item.flags


@pytest.mark.parametrize("owner", [None, "", "we", "Team", "everyone", "TBD", "Unassigned", "N/A", "I"])
def test_generic_or_missing_owner_means_no_owner_without_a_flag(v, owner):
    (item,), _ = verify_items("action_item", [action(owner=owner)], v)
    assert item.owner is None and item.flags == []


def test_owner_that_is_not_text_is_cleared(v):
    (item,), _ = verify_items("action_item", [action(owner=["Sam"])], v)
    assert item.owner is None and item.flags


def test_a_name_only_mentioned_in_the_transcript_is_accepted():
    turns = parse_transcript("Dana: Please ask Marcus about the budget for next quarter.")[0]
    (item,), _ = verify_items("action_item", [{"owner": "Marcus", "task": "ask about budget", "due": None,
                                               "quote": "Please ask Marcus about the budget"}], Verifier(turns))
    assert item.owner == "Marcus"


# ---- due dates --------------------------------------------------------------------------------
def test_due_date_stated_in_transcript_is_kept_verbatim(v):
    (item,), _ = verify_items("action_item", [action(due="Friday")], v)
    assert item.due == "Friday"


@pytest.mark.parametrize("due", ["2024-06-07", "next Tuesday", "end of month"])
def test_due_date_not_in_transcript_is_cleared_and_flagged(v, due):
    (item,), _ = verify_items("action_item", [action(due=due)], v)
    assert item.due is None and any("due date" in f for f in item.flags)


@pytest.mark.parametrize("due", [None, "", "none", "N/A", "not stated", "TBD"])
def test_no_due_date_is_fine(v, due):
    (item,), _ = verify_items("action_item", [action(due=due)], v)
    assert item.due is None and item.flags == []


# ---- decisions, questions, malformed ---------------------------------------------------------------
def test_decision_and_question_need_quotes_too(v):
    ok_d, _ = verify_items("decision", [{"text": "Launch on March 15", "quote": "we decided to launch on March 15"}], v)
    bad_d, rej = verify_items("decision", [{"text": "Hire two people", "quote": "we decided to hire two people"}], v)
    ok_q, _ = verify_items("open_question", [{"text": "Who owns billing?", "quote": "Does anyone know who owns the billing migration"}], v)
    assert len(ok_d) == 1 and bad_d == [] and len(rej) == 1 and len(ok_q) == 1
    assert ok_d[0].speaker == "Priya"


@pytest.mark.parametrize("raw", [None, "text", 5, [], {"task": "x"}, {"task": "", "quote": "I will update the release notes"},
                                 {"task": 5, "quote": "I will update the release notes"}])
def test_malformed_items_are_rejected_not_crashing(v, raw):
    accepted, rejected = verify_items("action_item", [raw], v)
    assert accepted == [] and len(rejected) == 1


def test_non_list_is_ignored(v):
    assert verify_items("decision", None, v) == ([], [])
    assert verify_items("decision", "nope", v) == ([], [])


def test_dedupe_by_quote_and_by_owner_task_and_keeps_order(v):
    a, _ = verify_items("action_item", [action(), action(task="Update notes again"), action(owner="Sam", quote="Priya, please confirm the QA schedule with the vendor", task="Confirm QA")], v)
    out = dedupe(a + a)
    assert [i.turn_index for i in out] == sorted(i.turn_index for i in out)
    assert len(out) == 2  # the repeated Friday quote collapses; the QA item stays


def test_norm():
    assert norm("  Hello,   WORLD! ’s ") == "hello world s"


@pytest.mark.parametrize("owner", ["the team", "The Team", "an owner to be decided", "a person"])
def test_articles_do_not_make_a_generic_owner_specific(v, owner):
    (item,), _ = verify_items("action_item", [action(owner=owner)], v)
    assert item.owner is None and (item.flags == [] or "not found" in item.flags[0])
