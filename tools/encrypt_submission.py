#!/usr/bin/env python3
"""Encrypt an ISU-Challenge submission for its pull request.

Pull requests are public, so a submission is committed encrypted with the organizers'
age public key; only the evaluation runner can decrypt it. Keep the plaintext submission
outside submissions/, for example in private/track_b/ (ignored by Git), and run from the
repository root:

    python tools/encrypt_submission.py --source private/track_b --output submissions/<team_id>/track_b

This writes submission.tar.age.000, .001, ... (each below 90 MiB) into the output folder,
replacing older parts. Commit only these files.

Requires the age tool (https://github.com/FiloSottile/age#installation):
macOS `brew install age`, Ubuntu/Debian `sudo apt install age`, Windows `winget install FiloSottile.age`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import NoReturn

# Public key of the organizers. Submissions encrypted with another key cannot be evaluated.
ORGANIZER_PUBLIC_KEY = "age126thdfpyav5wwrffmgnl08lukw5hmedan0auqm966x44up2kwf3qzfx329"
PART_BYTES = 90 * 1024 * 1024
# The runner accepts 2048 MB of parts; this leaves room for tar headers and encryption overhead.
MAX_BYTES = 2000 * 1024 * 1024
MAX_FILES = 20_000
SKIPPED = {".DS_Store", "Thumbs.db", "__pycache__", ".ipynb_checkpoints", ".git"}
TRACKS = ("track_a", "track_b")


def fail(message: str) -> NoReturn:
    sys.exit(f"error: {message}")


def collect_files(source: Path) -> list[Path]:
    files = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in SKIPPED for part in relative.parts) or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            fail(f"symbolic links are not allowed: {relative}")
        if path.is_file():
            files.append(path)
    return files


def check_layout(track: str, source: Path) -> None:
    if track == "track_a":
        if not (source / "generated").is_dir() or not (source / "generation_times.json").is_file():
            fail(f"a Track A submission needs generated/ and generation_times.json in {source}")
    elif not (source / "system.py").is_file():
        fail(f"a Track B submission needs system.py in {source}")


def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.mode & 0o111 else 0o644
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Plaintext submission folder, e.g. private/track_b.")
    parser.add_argument("--output", type=Path, required=True, help="Team folder, e.g. submissions/team-07/track_b.")
    parser.add_argument("--recipient", default=ORGANIZER_PUBLIC_KEY, help=argparse.SUPPRESS)
    args = parser.parse_args()

    source, output = args.source.resolve(), args.output.resolve()
    if args.recipient.startswith("age1REPLACE"):
        fail("the organizer public key is not published yet; see the README")
    if output.name not in TRACKS or output.parent.parent.name != "submissions":
        fail("--output must be submissions/<team_id>/track_a or submissions/<team_id>/track_b")
    if not source.is_dir():
        fail(f"source folder not found: {args.source}")
    if source == output.parent.parent or output.parent.parent in source.parents:
        fail("keep the plaintext submission outside submissions/, e.g. in private/ (ignored by Git)")
    check_layout(output.name, source)

    files = collect_files(source)
    total = sum(path.stat().st_size for path in files)
    if not files:
        fail(f"{args.source} contains no files")
    if len(files) > MAX_FILES or total > MAX_BYTES:
        fail(f"the submission has {len(files)} files and {total / 1024**2:.0f} MB; "
             f"the limits are {MAX_FILES} files and {MAX_BYTES // 1024**2} MB")
    if output.is_dir():
        others = [path for path in output.rglob("*") if path.is_file() and not path.name.startswith("submission.tar.age.")]
        if others:
            fail(f"{args.output} may only contain the encrypted parts; move these files elsewhere: "
                 + ", ".join(str(path.relative_to(output)) for path in others[:5]))

    with tempfile.TemporaryDirectory(prefix="isu-submission-") as temp:
        archive, encrypted = Path(temp) / "submission.tar", Path(temp) / "submission.tar.age"
        with tarfile.open(archive, "w") as tar:
            for path in files:
                tar.add(path, arcname=path.relative_to(source).as_posix(), recursive=False, filter=normalized)
        try:
            completed = subprocess.run(["age", "--encrypt", "--recipient", args.recipient,
                                        "--output", str(encrypted), str(archive)])
        except FileNotFoundError:
            fail("the age tool is not installed; see https://github.com/FiloSottile/age#installation")
        if completed.returncode != 0:
            fail("age could not encrypt the submission")

        output.mkdir(parents=True, exist_ok=True)
        for old in output.glob("submission.tar.age.*"):
            old.unlink()
        parts = []
        with encrypted.open("rb") as handle:
            while chunk := handle.read(PART_BYTES):
                part = output / f"submission.tar.age.{len(parts):03d}"
                part.write_bytes(chunk)
                parts.append(part)

    print(f"Encrypted {len(files)} files ({total / 1024**2:.1f} MB) into {len(parts)} part(s) in {args.output}")
    print(f"Next: git add {args.output} && git commit -m \"Submission\" && git push")


if __name__ == "__main__":
    main()
