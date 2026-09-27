"""Flag long exact phrase overlap between the thesis and local reference PDFs.

This is a review aid, not a substitute for an institution-approved similarity service.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "reports" / "thesis" / "main.pdf"
REFERENCE_DIR = ROOT / "references" / "papers"
OUTPUT = ROOT / "outputs" / "tables" / "originality_audit.json"
SHINGLE_WORDS = 14


def pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def shingles(words: list[str]) -> dict[tuple[str, ...], int]:
    return {tuple(words[index : index + SHINGLE_WORDS]): index for index in range(len(words) - SHINGLE_WORDS + 1)}


def main() -> None:
    thesis_text = pdf_text(THESIS)
    thesis_body = thesis_text.rsplit("References", 1)[0]
    thesis_words = tokens(thesis_body)
    thesis_shingles = shingles(thesis_words)
    matches: dict[str, list[str]] = defaultdict(list)

    for reference in sorted(REFERENCE_DIR.glob("*.pdf")):
        reference_shingles = shingles(tokens(pdf_text(reference)))
        shared = set(thesis_shingles).intersection(reference_shingles)
        for shingle in sorted(shared):
            phrase = " ".join(shingle)
            if phrase not in matches[reference.name]:
                matches[reference.name].append(phrase)

    report = {
        "scope": "Exact normalized phrase comparison against local reference PDFs only",
        "institutional_similarity_check_still_required": True,
        "minimum_exact_phrase_words": SHINGLE_WORDS,
        "thesis_body_word_count": len(thesis_words),
        "reference_pdf_count": len(list(REFERENCE_DIR.glob("*.pdf"))),
        "files_with_matches": len(matches),
        "matches": matches,
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
