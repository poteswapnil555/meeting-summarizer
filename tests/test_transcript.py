import pytest

from meeting_summarizer.transcript import fmt_time, parse_transcript


def parse(text, fmt="auto"):
    return parse_transcript(text, fmt)


# ---- plain text -------------------------------------------------------------------------
def test_plain_speaker_lines():
    turns, warnings = parse("Dana: Let's start.\nSam: Sounds good.")
    assert [(t.speaker, t.text, t.start) for t in turns] == [("Dana", "Let's start.", None), ("Sam", "Sounds good.", None)]
    assert [t.index for t in turns] == [0, 1] and warnings == []


@pytest.mark.parametrize("line,start", [
    ("[00:01:02] Dana: hello there", 62.0),
    ("(01:02) Dana: hello there", 62.0),
    ("[1:02:03] Dana: hello there", 3723.0),
    ("00:01:02 Dana: hello there", 62.0),
    ("[00:01:02] - Dana: hello there", 62.0),
])
def test_plain_timestamps(line, start):
    (t,), _ = parse(line)
    assert (t.speaker, t.text, t.start) == ("Dana", "hello there", start)


def test_continuation_lines_join_the_previous_turn():
    turns, _ = parse("Dana: first part\nsecond part\n\nthird part\nSam: reply")
    assert [t.text for t in turns] == ["first part second part third part", "reply"]


def test_text_without_any_speaker_is_one_anonymous_turn_per_start():
    turns, _ = parse("just some notes\nmore notes")
    assert len(turns) == 1 and turns[0].speaker is None and turns[0].text == "just some notes more notes"


def test_multi_word_speaker_names():
    (t,), _ = parse("Dr. Mary Jane O'Neil: hello")
    assert t.speaker == "Dr. Mary Jane O'Neil"


def test_urls_and_times_are_not_speakers():
    turns, _ = parse("Dana: see http://example.com/page and meet at 10:30 tomorrow")
    assert len(turns) == 1 and "http://example.com/page" in turns[0].text


def test_crlf_blank_and_whitespace_lines():
    turns, _ = parse("Dana: one\r\n\r\n   \r\nSam: two\r\n")
    assert [t.speaker for t in turns] == ["Dana", "Sam"]


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n", None, 42, [], b"bytes"])
def test_empty_or_non_text_input_gives_no_turns(text):
    assert parse(text) == ([], [])


def test_null_characters_are_removed():
    (t,), _ = parse("Dana: he\x00llo")
    assert t.text == "hello"


def test_speaker_with_empty_text_is_dropped():
    turns, _ = parse("Dana:\nSam: hi")
    assert [t.speaker for t in turns] == ["Sam"]


# ---- WebVTT -------------------------------------------------------------------------------
VTT = """WEBVTT

NOTE this is a comment
that spans two lines

STYLE
::cue { color: red }

1
00:00:01.000 --> 00:00:03.500 align:start position:0%
<v Dana>Let's start the meeting.</v>

2
00:00:04.000 --> 00:00:06.000
<v Dana>We have three topics.</v>

intro-cue
00:00:07.250 --> 00:00:09.000
<v Sam Lee><i>Sounds</i> good &amp; ready.</v>
"""


def test_vtt_voice_tags_merge_consecutive_cues_and_skip_notes():
    turns, warnings = parse(VTT)
    assert [(t.speaker, t.text) for t in turns] == [
        ("Dana", "Let's start the meeting. We have three topics."),
        ("Sam Lee", "Sounds good & ready."),
    ]
    assert turns[0].start == 1.0 and turns[1].start == 7.25 and warnings == []


def test_vtt_autodetect_by_header_and_by_cue_lines():
    assert parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nDana: hi\n")[0][0].speaker == "Dana"
    assert parse("00:00:01.000 --> 00:00:02.000\nDana: hi\n")[0][0].speaker == "Dana"  # header missing


def test_vtt_hourless_timestamps_and_comma_decimals():
    turns, _ = parse("WEBVTT\n\n01:05.500 --> 01:07.000\nDana: hi\n\n00:00:09,250 --> 00:00:10,000\nSam: yo\n")
    assert [t.start for t in turns] == [65.5, 9.25]


def test_vtt_bom_and_crlf():
    turns, _ = parse("﻿WEBVTT\r\n\r\n00:00:01.000 --> 00:00:02.000\r\nDana: hi\r\n")
    assert len(turns) == 1 and turns[0].text == "hi"


def test_vtt_multi_line_cue_text():
    turns, _ = parse("WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Dana>first line\nsecond line</v>\n")
    assert turns[0].text == "first line second line"


def test_vtt_cue_without_speaker_has_none_speaker():
    turns, _ = parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\njust words here\n")
    assert turns[0].speaker is None and turns[0].text == "just words here"


def test_vtt_unreadable_timestamp_is_skipped_with_warning():
    turns, warnings = parse("WEBVTT\n\n00:xx:01.000 --> 00:00:02.000\nDana: lost\n\n00:00:03.000 --> 00:00:04.000\nSam: kept\n")
    assert [t.speaker for t in turns] == ["Sam"] and len(warnings) == 1


def test_vtt_empty_cues_and_header_only():
    assert parse("WEBVTT\n") == ([], [])
    turns, _ = parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\n\n00:00:03.000 --> 00:00:04.000\n<v Dana></v>\n")
    assert turns == []


def test_vtt_different_speakers_are_not_merged():
    turns, _ = parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v A>one</v>\n\n00:00:02.000 --> 00:00:03.000\n<v B>two</v>\n\n00:00:03.000 --> 00:00:04.000\n<v A>three</v>\n")
    assert [t.speaker for t in turns] == ["A", "B", "A"]


def test_vtt_class_voice_tag():
    turns, _ = parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v.loud Dana>hello</v>\n")
    assert turns[0].speaker == "Dana"


def test_forced_format_overrides_detection():
    turns, _ = parse("00:00:01.000 --> 00:00:02.000\nDana: hi", fmt="text")
    assert turns[0].text.startswith("00:00:01.000 -->") and turns[1].speaker == "Dana"  # read as plain text, not as a cue


def test_fmt_time():
    assert fmt_time(None) == "" and fmt_time(62) == "00:01:02" and fmt_time(3723.9) == "01:02:03"
