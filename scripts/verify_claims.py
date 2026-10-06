"""Check that every accuracy/F1 percentage in the deck and paper matches a raw
value in results/comparison/*.csv.

Only direct percentage figures ("84.81%") are checked against literal CSV
cells. Point differences ("+7.5 points") and ratios ("1.39x") are computed
between cells rather than copied from one, so they are checked by hand.
Figures quoted from the published literature rather than measured here are
listed in LITERATURE_ALLOWLIST.

Usage: python scripts/verify_claims.py deck.pptx paper.md
"""
import csv
import re
import subprocess
import sys
import zipfile
from pathlib import Path

COMPARISON_DIR = Path(__file__).resolve().parent.parent / "results" / "comparison"

# Figures self-reported by the base TrafficCLIP paper and by the
# OptimizedTrafficCLIP manuscript. Not measured by this project.
LITERATURE_ALLOWLIST = {
    "6.79%", "47.88%", "70%", "15.5-point", "15.5pt", "1.39", "1.39×", "39%",
    "~39%", "0.9",
}


def percent_values_from_csvs():
    values = set()
    for path in COMPARISON_DIR.glob("*.csv"):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                for key, val in row.items():
                    if val is None:
                        continue
                    try:
                        f_val = float(val)
                    except ValueError:
                        continue
                    if 0.0 <= f_val <= 1.0:
                        # stored as a fraction (accuracy/f1/precision/recall)
                        for nd in (1, 2):
                            values.add(f"{f_val * 100:.{nd}f}")
    return values


def extract_text(path):
    if path.suffix == ".pptx":
        with zipfile.ZipFile(path) as z:
            texts = []
            for name in z.namelist():
                if re.match(r"ppt/slides/slide\d+\.xml", name):
                    xml = z.read(name).decode("utf-8")
                    texts.extend(re.findall(r"<a:t>(.*?)</a:t>", xml, re.DOTALL))
        return "\n".join(texts)
    return path.read_text()


PERCENT_RE = re.compile(r"(\d{1,3}\.\d{1,2})\s*%")


def main():
    if len(sys.argv) != 3:
        print("usage: python scripts/verify_claims.py deck.pptx paper.md")
        sys.exit(2)

    gold = percent_values_from_csvs()
    problems = []

    for arg in sys.argv[1:]:
        path = Path(arg)
        text = extract_text(path)
        for match in PERCENT_RE.finditer(text):
            num = match.group(1)
            token = match.group(0).replace(" ", "")
            if token in LITERATURE_ALLOWLIST or num in LITERATURE_ALLOWLIST:
                continue
            if num not in gold:
                context = text[max(0, match.start() - 40):match.end() + 10].replace("\n", " ")
                problems.append(f"{path.name}: {token!r} not found in any results/comparison/*.csv cell -- context: ...{context}...")

    if problems:
        print(f"FAILED - {len(problems)} unverified percentage claim(s):")
        for p in problems:
            print(" -", p)
        sys.exit(1)

    print("All percentage claims verified against results/comparison/*.csv")


if __name__ == "__main__":
    main()
