"""Build submission.zip with only the source and small assets needed to reproduce the verified result."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE_DIRS = ["duck_kick", "tests", "scripts", "tools", "evidence"]
INCLUDE_FILES = ["README.md", "LICENSE", "NOTICE", "requirements.txt", ".gitignore"]
EXCLUDE_REL = set()
EXCLUDE_SUFFIXES = {".pyc", ".log", ".mp4", ".zip", ".gif"}
EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_PARTS = {"__pycache__", ".git", ".venv", "runs"}


def main(out: Path = ROOT / "submission.zip", checkpoint: str = "duck_kick/checkpoints/final.pt") -> int:
    files = [ROOT / f for f in INCLUDE_FILES]
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and p.suffix not in EXCLUDE_SUFFIXES and p.name not in EXCLUDE_NAMES and not (set(p.parts) & EXCLUDE_PARTS) and p.relative_to(ROOT).as_posix() not in EXCLUDE_REL:
                if "checkpoints" in p.parts and p != ROOT / checkpoint and p.name not in ("training_evidence.json", "celebration.pt", "walk.onnx", "approach.pt", "match_kick.pt"):
                    continue   # only the inference checkpoints needed by the verified runs ship
                files.append(p)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    total = sum(f.stat().st_size for f in files)
    print(f"{out.name}: {len(files)} files, {total/1e6:.2f} MB uncompressed, {out.stat().st_size/1e6:.2f} MB zipped")
    for f in files:
        print(f"  {f.stat().st_size:9d}  {f.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
