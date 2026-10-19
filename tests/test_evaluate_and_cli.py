import json
from pathlib import Path

import pytest

from conftest import RouterLLM, SequenceLLM, reply
from meeting_summarizer import __main__ as cli
from meeting_summarizer.chunker import chunk_turns
from meeting_summarizer.evaluate import MockLLM, evaluate, keywords_present, load_gold, main, matches
from meeting_summarizer.summarizer import summarize
from meeting_summarizer.transcript import parse_transcript
from meeting_summarizer.verify import Item, Verifier

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"
ENV_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_TENANT_ID",
            "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]


@pytest.fixture
def no_env(monkeypatch):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


GOLD = load_gold(EVAL / "gold.json")


# ---- the hand-written data is self-consistent ---------------------------------------------------------
def test_gold_has_five_transcripts_including_one_without_items():
    assert [g["id"] for g in GOLD] == ["t1", "t2", "t3", "t4", "t5"]
    t4 = next(g for g in GOLD if g["id"] == "t4")
    assert t4["expect_no_items"] and not t4["decisions"] and not t4["action_items"]


@pytest.mark.parametrize("g", GOLD, ids=[g["id"] for g in GOLD])
def test_every_gold_quote_really_is_in_its_transcript_and_owners_exist(g):
    turns, _ = parse_transcript((EVAL / g["file"]).read_text(encoding="utf-8"))
    assert turns
    v = Verifier(turns)
    for key in ("decisions", "action_items", "open_questions"):
        for item in g[key]:
            assert v.find_quote(item["quote"])[0] is not None, item["quote"]
    for a in g["action_items"]:
        assert v.check_owner(a["owner"]) == (a["owner"], None)
        if a["due"]:
            assert v.check_due(a["due"])[0] is not None, a["due"]


@pytest.mark.parametrize("g", GOLD, ids=[g["id"] for g in GOLD])
def test_gold_keywords_match_the_gold_text(g):
    for key in ("decisions", "action_items", "open_questions"):
        for item in g[key]:
            assert keywords_present(item.get("text") or item["task"], item["keywords"]), item


def test_t5_is_long_enough_to_need_several_chunks_and_t2_is_vtt():
    t5 = next(g for g in GOLD if g["id"] == "t5")
    turns, _ = parse_transcript((EVAL / t5["file"]).read_text(encoding="utf-8"))
    assert len(chunk_turns(turns, t5["chunk_chars"])) >= 2
    vtt, _ = parse_transcript((EVAL / "transcripts/t2_product_sync.vtt").read_text(encoding="utf-8"))
    assert {t.speaker for t in vtt} == {"Priya", "Tom", "Aisha"}


def test_bad_gold_files_rejected(tmp_path):
    f = tmp_path / "g.json"
    f.write_text("nope")
    with pytest.raises(ValueError, match="cannot read"):
        load_gold(f)
    f.write_text(json.dumps([{"id": "x", "file": "x.txt", "decisions": [{"text": "t"}]}]))
    with pytest.raises(ValueError, match="keywords"):
        load_gold(f)
    f.write_text(json.dumps([{"id": "x"}]))
    with pytest.raises(ValueError, match="id and file"):
        load_gold(f)


# ---- scoring rules -----------------------------------------------------------------------------------------
def item(text, owner=None, due=None):
    return Item("action_item", text, "q", None, "", 0, owner=owner, due=due)


def test_matching_rules():
    g = {"owner": "Lena", "keywords": ["fix", "checkout"], "due": "Wednesday"}
    assert matches(g, item("Write a fix for the Checkout bug", "Lena", "by Wednesday"), "action_item")
    assert not matches(g, item("Write a fix for the Checkout bug", "Omar", "Wednesday"), "action_item")  # wrong owner
    assert not matches(g, item("Write a fix for the Checkout bug", None, "Wednesday"), "action_item")  # no owner
    assert not matches(g, item("Write a fix for the Checkout bug", "Lena", None), "action_item")  # due missing
    assert not matches(g, item("Fix the login page", "Lena", "Wednesday"), "action_item")  # wrong task
    assert matches({"owner": "Omar", "keywords": ["review"], "due": None}, item("Review the PR", "Omar", "Friday"), "action_item")
    assert keywords_present("Free", ["free|discount"]) and not keywords_present("paid", ["free|discount"])


# ---- evaluation runs ----------------------------------------------------------------------------------------------
def run(llm, hook=None, **kw):
    return evaluate(llm, GOLD, EVAL, hook=hook, **kw)


def test_mock_evaluation_plumbing():
    llm = MockLLM()
    r = run(llm, hook=lambda g: setattr(llm, "current", g))
    r.mode = "mock"
    s = r.summary()
    assert s["transcripts"] == 5 and s["transcripts_status_ok"] == 5
    assert (s["decisions"]["found"], s["decisions"]["gold"]) == (7, 7) and s["decisions"]["extra_not_in_gold"] == 0
    assert (s["action_items"]["found"], s["action_items"]["gold"]) == (12, 12)
    assert (s["open_questions"]["found"], s["open_questions"]["gold"]) == (4, 4)
    assert s["no_item_transcripts_with_zero_items"] == "1/1" and s["proposed_items_rejected_by_code"] >= 2


def test_a_model_that_finds_nothing_scores_zero_and_invents_nothing():
    s = run(RouterLLM(lambda u: reply("Nothing."))).summary()
    assert s["action_items"]["found"] == 0 and s["action_items"]["extra_not_in_gold"] == 0
    assert s["no_item_transcripts_with_zero_items"] == "1/1"


def test_a_model_that_invents_everything_keeps_nothing():
    invented = reply("x", [{"text": "Buy a boat", "quote": "we decided to buy a boat for the team"}],
                     [{"owner": "Maya", "task": "Buy a boat", "due": "Friday", "quote": "I will buy a boat by Friday"}],
                     [{"text": "Why?", "quote": "why did nobody buy the boat"}])
    r = run(RouterLLM(lambda u: invented))
    s = r.summary()
    assert s["decisions"]["found"] == 0 and s["decisions"]["extra_not_in_gold"] == 0 and s["action_items"]["extra_not_in_gold"] == 0
    assert s["proposed_items_rejected_by_code"] >= 15 and s["no_item_transcripts_with_zero_items"] == "1/1"


def test_a_real_but_unlisted_item_is_counted_as_extra_not_as_wrong():
    extra = {"owner": "Omar", "task": "Think about priorities", "due": None, "quote": "I think it is the top priority"}
    r = run(RouterLLM(lambda u: reply("s", actions=[extra]) if "checkout bug is blocking" in u else reply("s")))
    assert r.counts["action_items"]["extra"] == 1


def test_wrong_owner_is_not_a_match():
    swapped = {"owner": "Omar", "task": "Write a fix for the checkout bug", "due": "Wednesday", "quote": "I will write a fix for the checkout bug by Wednesday"}
    r = run(RouterLLM(lambda u: reply("s", actions=[swapped]) if "write a fix" in u else reply("s")))
    assert r.counts["action_items"]["found"] == 0 and r.counts["action_items"]["extra"] == 1


def test_llm_errors_are_counted_as_failed_transcripts_not_crashes():
    r = run(SequenceLLM(TimeoutError("down")))
    assert r.transcripts == 5 and r.transcripts_ok == 0 and r.counts["action_items"]["found"] == 0


def test_delay_and_progress():
    slept, seen = [], []
    run(RouterLLM(lambda u: reply()), delay=1.5, sleep=slept.append, progress=lambda d, t: seen.append((d, t)))
    assert seen[0] == (1, 5) and seen[-1] == (5, 5) and slept == [1.5] * 4


def test_t5_uses_the_chunk_override_and_the_others_use_the_default():
    llm = RouterLLM(lambda u: reply())
    r = run(llm)
    chunks = {d["id"]: d["chunks"] for d in r.details}
    assert chunks["t5"] >= 2 and chunks["t1"] == chunks["t2"] == chunks["t3"] == chunks["t4"] == 1


# ---- evaluate CLI ------------------------------------------------------------------------------------------------------
def test_cli_mock_prints_mode_and_not_accuracy(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main(["--mock"]) == 0
    cap = capsys.readouterr()
    assert "MODE: mock" in cap.out and "NOT accuracy" in cap.out and "5/5 done" in cap.err


def test_cli_saves_details(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "e.json"
    assert main(["--mock", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data["details"]) == 5 and {"id", "status", "kinds", "rejected", "llm_calls"} <= set(data["details"][0])


def test_live_without_env_fails_clearly(no_env, capsys):
    assert main(["--live", "--gold", str(EVAL / "gold.json")]) == 2
    assert "No LLM configured" in capsys.readouterr().err


def test_live_with_missing_model_variable(no_env, capsys):
    no_env.setenv("OPENAI_API_KEY", "fake")
    assert main(["--live"]) == 2
    assert "OPENAI_MODEL is not set" in capsys.readouterr().err


def test_missing_gold_or_transcript_file(capsys, tmp_path):
    assert main(["--mock", "--gold", str(tmp_path / "nope.json")]) == 2
    g = tmp_path / "g.json"
    g.write_text(json.dumps([{"id": "x", "file": "missing.txt"}]))
    assert main(["--mock", "--gold", str(g)]) == 2
    assert "not found" in capsys.readouterr().err


# ---- summariser CLI ------------------------------------------------------------------------------------------------------
@pytest.fixture
def transcript_file(tmp_path):
    f = tmp_path / "meeting.txt"
    f.write_text((EVAL / "transcripts/t1_sprint_planning.txt").read_text(encoding="utf-8"), encoding="utf-8")
    return f


def test_cli_without_env(no_env, capsys, transcript_file):
    assert cli.main([str(transcript_file)]) == 2
    assert "No LLM configured" in capsys.readouterr().err


def test_cli_missing_file(no_env, capsys, tmp_path):
    assert cli.main([str(tmp_path / "nope.txt")]) == 2
    assert "Cannot read" in capsys.readouterr().err


def test_cli_prints_sections_with_quotes_and_drops(monkeypatch, capsys, transcript_file):
    good = {"owner": "Lena", "task": "Write a fix for the checkout bug", "due": "Wednesday", "quote": "I will write a fix for the checkout bug by Wednesday"}
    bad = {"owner": "Lena", "task": "Fly to the moon", "due": None, "quote": "Lena will fly to the moon next week"}
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda: RouterLLM(lambda u: reply("Sprint planning.", actions=[good, bad])))
    assert cli.main([str(transcript_file)]) == 0
    out = capsys.readouterr().out
    assert "SUMMARY\nSprint planning." in out and "- Lena: Write a fix for the checkout bug (due: Wednesday)" in out
    assert '[00:01:02] Lena: "I will write a fix for the checkout bug by Wednesday"' in out
    assert "DROPPED: 1 item(s)" in out and "Fly to the moon" not in out


def test_cli_json_output(monkeypatch, capsys, transcript_file):
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda: RouterLLM(lambda u: reply("S.")))
    assert cli.main([str(transcript_file), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "ok" and data["summary"] == "S." and data["action_items"] == []


def test_cli_empty_file_exits_nonzero_with_message(monkeypatch, capsys, tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda: RouterLLM(lambda u: reply()))
    assert cli.main([str(f)]) == 1
    assert "empty_transcript" in capsys.readouterr().err


def test_cli_llm_failure_exits_nonzero_without_traceback(monkeypatch, capsys, transcript_file):
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda: SequenceLLM(TimeoutError("down")))
    assert cli.main([str(transcript_file)]) == 1
    assert "failed" in capsys.readouterr().err


def test_cli_bad_environment_value_is_a_clear_message(monkeypatch, capsys, transcript_file):
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda: RouterLLM(lambda u: reply()))
    monkeypatch.setenv("CHUNK_CHARS", "lots")
    assert cli.main([str(transcript_file)]) == 2
    assert "CHUNK_CHARS must be a whole number" in capsys.readouterr().err


def test_cli_vtt_by_extension_content(monkeypatch, capsys, tmp_path):
    f = tmp_path / "m.vtt"
    f.write_text((EVAL / "transcripts/t2_product_sync.vtt").read_text(encoding="utf-8"), encoding="utf-8")
    seen = []
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda: RouterLLM(lambda u: seen.append(u) or reply("S.")))
    assert cli.main([str(f)]) == 0
    assert "[00:00:09] Tom:" in seen[0]
