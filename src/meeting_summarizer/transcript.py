"""Parse a meeting transcript (plain text or WebVTT) into speaker turns. Never raises on odd content.

Plain text understood:        Alice: text            [00:01:02] Alice: text         (01:02) Alice: text
                              00:01:02 Alice: text   (a line with no speaker continues the previous turn)
WebVTT understood:            cue ids, NOTE/STYLE/REGION blocks, cue settings, <v Name>text</v> voice tags,
                              "Name: text" prefixes, other tags stripped, HTML entities, hour-less timestamps,
                              CRLF line endings, a leading BOM, comma decimals; consecutive cues by one speaker are merged.
Known limit: in plain text, any line that looks like "Some Words: text" (up to 4 words) is read as a new speaker.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    index: int
    speaker: str | None
    start: float | None  # seconds from the start of the meeting, if the transcript has timestamps
    text: str


_TS = r"(?:(\d{1,3}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_CUE_RE = re.compile(rf"^\s*{_TS}\s*-->\s*{_TS}")
_VOICE_RE = re.compile(r"<v(?:\.[^\s>]+)*\s+([^>]+)>")
_TAG_RE = re.compile(r"<[^>]+>")
_LABEL = r"[A-Za-z][A-Za-z0-9_.'\-]*(?: [A-Za-z0-9_.'\-]+){0,3}"
_SPEAKER_RE = re.compile(rf"^(?P<spk>{_LABEL}):(?:\s+|$)(?P<text>.*)$")
_BRACKET_TS_RE = re.compile(r"^\s*[\[(](\d{1,3}:\d{2}(?::\d{2})?)[\])]\s*[-–]?\s*(?P<rest>.*)$")
_BARE_TS_RE = re.compile(rf"^\s*(\d{{1,3}}:\d{{2}}(?::\d{{2}})?)\s*[-–]?\s+(?P<rest>{_LABEL}:(?:\s+.*|$))$")


def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _clock_to_seconds(clock: str) -> float:
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _vtt_seconds(m: re.Match, offset: int = 1) -> float:
    h, mi, s, ms = m.group(offset), m.group(offset + 1), m.group(offset + 2), m.group(offset + 3)
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + (int((ms or "0").ljust(3, "0")) / 1000)


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()


def _split_speaker(text: str) -> tuple[str | None, str]:
    m = _SPEAKER_RE.match(text)
    if m:
        return m.group("spk").strip(), clean_text(m.group("text"))
    return None, clean_text(text)


def looks_like_vtt(text: str) -> bool:
    body = text.lstrip("﻿ \t\r\n")
    if body.startswith("WEBVTT"):
        return True
    return any(_CUE_RE.match(line) for line in body.splitlines()[:200])


def _merge(raw: list[tuple[str | None, float | None, str]]) -> list[Turn]:
    turns: list[Turn] = []
    for speaker, start, text in raw:
        if not text:
            continue
        if turns and speaker is not None and turns[-1].speaker == speaker:
            prev = turns[-1]
            turns[-1] = Turn(prev.index, prev.speaker, prev.start, f"{prev.text} {text}")
        else:
            turns.append(Turn(len(turns), speaker, start, text))
    return turns


def parse_vtt(text: str) -> tuple[list[Turn], list[str]]:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    raw, warnings = [], []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        cue_at = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if cue_at is None:
            continue  # a stray block without a timing line
        m = _CUE_RE.match(lines[cue_at])
        if not m:
            warnings.append(f"skipped a cue with an unreadable timestamp: {lines[cue_at][:60]!r}")
            continue
        start = _vtt_seconds(m, 1)
        body = " ".join(lines[cue_at + 1:])
        voice = _VOICE_RE.search(body)
        speaker = clean_text(html.unescape(voice.group(1))) if voice else None
        body = html.unescape(_TAG_RE.sub("", body))
        if speaker is None:
            speaker, body = _split_speaker(body)
        raw.append((speaker or None, start, clean_text(body)))
    return _merge(raw), warnings


def parse_plain(text: str) -> list[Turn]:
    turns: list[Turn] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.replace("\x00", "").strip()
        if not line:
            continue
        start, rest = None, line
        m = _BRACKET_TS_RE.match(line) or _BARE_TS_RE.match(line)
        if m:
            start, rest = _clock_to_seconds(m.group(1)), m.group("rest")
        speaker, body = _split_speaker(rest)
        if speaker is None and start is None and turns:  # a line with no speaker continues the previous turn
            prev = turns[-1]
            turns[-1] = Turn(prev.index, prev.speaker, prev.start, f"{prev.text} {body}".strip())
        else:
            turns.append(Turn(len(turns), speaker, start, body))
    turns = [t for t in turns if t.text]
    return [Turn(i, t.speaker, t.start, t.text) for i, t in enumerate(turns)]


def parse_transcript(text: object, fmt: str = "auto") -> tuple[list[Turn], list[str]]:
    """Return (turns, warnings). fmt: 'auto', 'vtt' or 'text'. Empty or non-text input gives no turns."""
    if not isinstance(text, str) or not text.strip():
        return [], []
    use_vtt = fmt == "vtt" or (fmt == "auto" and looks_like_vtt(text))
    if use_vtt:
        return parse_vtt(text)
    return parse_plain(text), []
