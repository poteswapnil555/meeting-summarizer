import importlib.util
from pathlib import Path

import pytest

from meeting_summarizer.transcript import parse_transcript

ROOT = Path(__file__).resolve().parent.parent

# Invented text in the same layout as the real PDF (attendee list, labels with a period, page footers, curly quotes)
RAW = """Conference Call of the Committee on
January 1, 2000
PRESENT:
Jane Roe, Chair
John Doe, Vice Chair
CHAIR ROE.  Good morning.  Let’s begin.  We have one item today.
January 1, 2000
2 of 9
MR. DOE.  Thank you, Chair.  I don’t have much to add—but one thing
about the schedule.
MS. SMITH.  I agree.
VICE CHAIR LEE.  Noted.
"""


def _script():
    spec = importlib.util.spec_from_file_location("download_transcript", ROOT / "scripts" / "download_transcript.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_convert_drops_attendee_list_and_footers_and_keeps_punctuation():
    turns = _script().convert(RAW)
    assert turns == [
        ("Chair Roe", "Good morning. Let’s begin. We have one item today."),
        ("Mr. Doe", "Thank you, Chair. I don’t have much to add—but one thing about the schedule."),
        ("Ms. Smith", "I agree."),
        ("Vice Chair Lee", "Noted."),
    ]


def test_converted_text_parses_into_the_same_speakers():
    mod = _script()
    text = "\n".join(f"{s}: {b}" for s, b in mod.convert(RAW))
    turns, _ = parse_transcript(text)
    assert [t.speaker for t in turns] == ["Chair Roe", "Mr. Doe", "Ms. Smith", "Vice Chair Lee"]


def test_convert_of_text_without_speakers_is_empty():
    assert _script().convert("just a page of text\nwith no speaker labels") == []


class _Resp:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self.data


def test_download_writes_text_and_respects_max_turns(tmp_path, monkeypatch):
    mod = _script()
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"%PDF-1.7 fake"))
    monkeypatch.setattr(mod, "pdf_to_text", lambda data: RAW)
    dest = tmp_path / "demo" / "t.txt"
    assert mod.download("https://example.invalid/x.pdf", dest, max_turns=2) == 0
    assert dest.read_text(encoding="utf-8").splitlines()[0].startswith("Chair Roe: Good morning.")
    assert len(dest.read_text(encoding="utf-8").splitlines()) == 2


def test_download_rejects_non_pdf(tmp_path, monkeypatch, capsys):
    mod = _script()
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"<html>404</html>"))
    assert mod.download("https://example.invalid/x.pdf", tmp_path / "t.txt") == 1 and not (tmp_path / "t.txt").exists()
    assert "not a PDF" in capsys.readouterr().err


def test_download_network_error_is_clean(tmp_path, monkeypatch, capsys):
    mod = _script()

    def boom(req, timeout=0):
        raise mod.urllib.error.URLError("offline")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    assert mod.download("https://example.invalid/x.pdf", tmp_path / "t.txt") == 1
    assert "Download failed" in capsys.readouterr().err


def test_download_unreadable_pdf_is_a_clean_failure(tmp_path, monkeypatch, capsys):
    mod = _script()
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"%PDF-1.7 not really a pdf"))
    assert mod.download("https://example.invalid/x.pdf", tmp_path / "t.txt") == 1
    assert "Conversion failed" in capsys.readouterr().err


def test_download_pdf_without_speakers_is_a_clean_failure(tmp_path, monkeypatch, capsys):
    mod = _script()
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"%PDF-1.7 fake"))
    monkeypatch.setattr(mod, "pdf_to_text", lambda data: "no speakers here")
    assert mod.download("https://example.invalid/x.pdf", tmp_path / "t.txt") == 1
    assert "no speaker turns" in capsys.readouterr().err


def test_download_skips_existing(tmp_path, monkeypatch):
    mod = _script()
    dest = tmp_path / "t.txt"
    dest.write_text("old")
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not download"))
    assert mod.download("https://example.invalid/x.pdf", dest) == 0 and dest.read_text() == "old"
