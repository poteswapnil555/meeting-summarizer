import pytest

from meeting_summarizer.chunker import MIN_CHUNK_CHARS, chunk_turns, render_turn
from meeting_summarizer.transcript import Turn, parse_transcript


def turns_of(n, words=20, speaker="Dana"):
    return [Turn(i, f"{speaker}{i % 3}", i * 10.0, " ".join(f"w{i}x{j}" for j in range(words))) for i in range(n)]


def test_small_transcript_is_one_chunk():
    chunks = chunk_turns(turns_of(3), 6000)
    assert len(chunks) == 1 and chunks[0].turn_indices == (0, 1, 2)
    assert chunks[0].text.splitlines()[0].startswith("[00:00:00] Dana0: w0x0")


def test_turns_are_never_cut_when_they_fit():
    turns = turns_of(40)
    chunks = chunk_turns(turns, 600)
    assert len(chunks) > 5
    rendered = {render_turn(t) for t in turns}
    for c in chunks:
        assert len(c.text) <= 600
        for line in c.text.split("\n"):
            assert line in rendered  # every line is a complete, unmodified turn


def test_every_turn_lands_in_exactly_one_chunk_in_order():
    chunks = chunk_turns(turns_of(50), 700)
    seen = [i for c in chunks for i in c.turn_indices]
    assert seen == list(range(50)) and [c.index for c in chunks] == list(range(len(chunks)))


def test_very_long_transcript_makes_many_chunks():
    chunks = chunk_turns(turns_of(2000), 1000)
    assert len(chunks) > 100 and all(len(c.text) <= 1000 for c in chunks)


def test_single_huge_turn_is_split_into_bounded_pieces():
    sentence = "This is one sentence of the huge turn. "
    turn = Turn(0, "Dana", None, sentence * 500)
    chunks = chunk_turns([turn], 1000)
    assert len(chunks) > 10
    for c in chunks:
        assert len(c.text) <= 1000 and c.text.startswith("Dana: ") and c.turn_indices == (0,)
    # nothing lost: all sentences are still there
    joined = " ".join(c.text.removeprefix("Dana: ") for c in chunks)
    assert joined.count("This is one sentence of the huge turn.") == 500


def test_huge_turn_between_normal_turns_keeps_the_neighbours_whole():
    turns = [Turn(0, "A", None, "short one here"), Turn(1, "B", None, "word " * 1000), Turn(2, "C", None, "short two here")]
    chunks = chunk_turns(turns, 500)
    assert chunks[0].text == "A: short one here" and chunks[-1].text == "C: short two here"
    assert all(len(c.text) <= 500 for c in chunks)


def test_huge_turn_without_spaces_or_sentences_is_hard_split():
    chunks = chunk_turns([Turn(0, "Dana", None, "x" * 5000)], 500)
    assert len(chunks) >= 10 and all(len(c.text) <= 500 for c in chunks)


def test_turn_exactly_at_the_limit_is_not_split():
    t = Turn(0, "D", None, "a" * 294)
    line_len = len(render_turn(t))
    assert line_len >= MIN_CHUNK_CHARS
    chunks = chunk_turns([t], line_len)  # the rendered line is exactly the chunk limit
    assert len(chunks) == 1 and chunks[0].text == render_turn(t)


def test_empty_input_gives_no_chunks():
    assert chunk_turns([], 1000) == []


def test_too_small_chunk_size_rejected():
    with pytest.raises(ValueError):
        chunk_turns(turns_of(2), MIN_CHUNK_CHARS - 1)


def test_chunking_a_parsed_transcript_without_speakers_or_times():
    turns, _ = parse_transcript("just notes\n" * 3 + "Sam: hello")
    assert chunk_turns(turns, 1000)[0].text.splitlines()[-1] == "Sam: hello"
