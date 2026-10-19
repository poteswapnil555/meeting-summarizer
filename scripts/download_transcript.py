"""Download the FOMC conference call transcript of 4 October 2019 and convert it to plain text for the demo.

The transcript is a third-party document: it is NOT part of this repository (data/demo/ is git-ignored).
Source:  https://www.federalreserve.gov/monetarypolicy/files/FOMC20191004confcall.pdf
Licence: the Federal Reserve Board states that information on its website is in the public domain and may be copied and
         distributed without permission, and asks users to cite the Board as the source
         (https://www.federalreserve.gov/disclaimer.htm). Material from non-Board sources is excluded; check the notice yourself.

The PDF is long (75 pages, about 150,000 characters). The whole thing is roughly 26 chunks, which means 30+ LLM calls.
Use --max-turns N to keep only the first N speaker turns for a cheaper demo.

Usage:  python scripts/download_transcript.py [--force] [--max-turns N]
Output: data/demo/fomc_2019-10-04.txt, one "Speaker: text" line per turn.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://www.federalreserve.gov/monetarypolicy/files/FOMC20191004confcall.pdf"
DEST = Path("data/demo/fomc_2019-10-04.txt")
MAX_BYTES = 50 * 1024 * 1024

_LABEL = re.compile(r"^(?P<label>(?:VICE CHAIR|CHAIR)\s+[A-Z][A-Z'\-]+|M[RS]S?\.\s+[A-Z][A-Z'\-]+)\.\s+", re.MULTILINE)
_FOOTER = re.compile(r"^\s*[A-Z][a-z]+ \d{1,2}, \d{4}\s*\n\s*\d+ of \d+\s*$", re.MULTILINE)


def _title(label: str) -> str:
    return " ".join(w.capitalize() if not w.endswith(".") else w.capitalize() for w in label.split())


def convert(raw: str) -> list[tuple[str, str]]:
    """Turn the extracted PDF text into (speaker, text) turns. The attendee list before the first speaker is dropped."""
    text = _FOOTER.sub("", raw.replace("\r\n", "\n"))
    text = re.sub(r"�(?=(?:s|t|d|ll|m|re|ve)\b)", "'", text)  # the PDF's apostrophes come out as U+FFFD
    text = text.replace("�", " - ")
    matches = list(_LABEL.finditer(text))
    turns = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[m.end():end]).strip()
        if body:
            turns.append((_title(m.group("label")), body))
    return turns


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    return "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(data)).pages)


def download(url: str, dest: Path, force: bool = False, max_turns: int | None = None) -> int:
    if dest.exists() and not force:
        print(f"Already present: {dest} (use --force to download again)")
        return 0
    req = urllib.request.Request(url, headers={"User-Agent": "meeting-summarizer-download/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    if len(data) > MAX_BYTES or not data.startswith(b"%PDF"):
        print("Download failed: the response is not a PDF of a sensible size", file=sys.stderr)
        return 1
    try:
        turns = convert(pdf_to_text(data))
    except Exception as e:  # unreadable PDF
        print(f"Conversion failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if not turns:
        print("Conversion failed: no speaker turns were found in the PDF text", file=sys.stderr)
        return 1
    if max_turns is not None:
        turns = turns[:max(max_turns, 1)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(f"{speaker}: {body}" for speaker, body in turns) + "\n", encoding="utf-8")
    print(f"Saved {dest}: {len(turns)} turns. Credit: Federal Reserve Board, FOMC conference call transcript, October 4, 2019.")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    ap.add_argument("--max-turns", type=int, help="keep only the first N speaker turns")
    args = ap.parse_args(argv)
    return download(URL, DEST, args.force, args.max_turns)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
