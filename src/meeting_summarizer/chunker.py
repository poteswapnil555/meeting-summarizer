"""Split turns into chunks for the map step. A speaker turn is never cut unless one turn alone is larger than a chunk."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .transcript import Turn, fmt_time

MIN_CHUNK_CHARS = 200


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    turn_indices: tuple[int, ...]


def render_turn(turn: Turn, text: str | None = None) -> str:
    prefix = f"[{fmt_time(turn.start)}] " if turn.start is not None else ""
    who = f"{turn.speaker}: " if turn.speaker else ""
    return f"{prefix}{who}{turn.text if text is None else text}"


def _split_long(text: str, limit: int) -> list[str]:
    """Split one very long turn at sentence ends, then at spaces, then hard, into pieces of at most limit characters."""
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > limit:  # a single sentence longer than a chunk
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def chunk_turns(turns: list[Turn], max_chars: int = 6000) -> list[Chunk]:
    if max_chars < MIN_CHUNK_CHARS:
        raise ValueError(f"chunk size must be at least {MIN_CHUNK_CHARS} characters")
    chunks: list[Chunk] = []
    lines: list[str] = []
    indices: list[int] = []
    size = 0

    def flush() -> None:
        nonlocal lines, indices, size
        if lines:
            chunks.append(Chunk(len(chunks), "\n".join(lines), tuple(indices)))
        lines, indices, size = [], [], 0

    for turn in turns:
        line = render_turn(turn)
        if len(line) > max_chars:  # one huge turn: the only case where a turn is split
            flush()
            header = render_turn(turn, "")
            for piece in _split_long(turn.text, max(max_chars - len(header), 50)):
                chunks.append(Chunk(len(chunks), render_turn(turn, piece), (turn.index,)))
            continue
        if size and size + 1 + len(line) > max_chars:
            flush()
        lines.append(line)
        indices.append(turn.index)
        size += len(line) + (1 if size else 0)
    flush()
    return chunks
