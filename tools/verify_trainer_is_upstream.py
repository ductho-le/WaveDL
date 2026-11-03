"""Check that src/wavedl is upstream WaveDL plus the fix branches and nothing else.

Run from the repository root of a checkout that has the upstream remote and the
fix branches:

    python tools/verify_trainer_is_upstream.py

Every trainer file a branch touches is merged with ``git merge-file``, branch by
branch in order, onto its upstream content. Nothing in the checkout changes.
The working tree must then differ from upstream only in those files and match
each merge byte for byte outside its conflict regions. Branches that edit the
same lines conflict, and no merge can resolve that by itself, so each region
the working tree resolves by hand is printed for review instead.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


TRAINER = "src/wavedl"
UPSTREAM = "upstream/main"
BRANCHES = (
    "pr/small-fixes",
    "pr/cv-parity",
    "pr/onnx-export-safety",
    "pr/launcher-process-cleanup",
    "pr/metrics-jsonl",
)
# Long markers cannot collide with "=======" lines in upstream sources.
MARKER = 40


class Mismatch(Exception):
    pass


def git(*args: str, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(["git", *args], capture_output=True)
    if result.returncode not in ok:
        error = result.stderr.decode(errors="replace").strip()
        raise Mismatch(f"git {' '.join(args)} failed:\n{error}")
    return result


def names(*args: str) -> set[str]:
    return set(git(*args).stdout.decode().splitlines())


def content(rev: str, path: str) -> bytes | None:
    result = git("cat-file", "blob", f"{rev}:{path}", ok=(0, 128))
    return result.stdout if result.returncode == 0 else None


def merge_file(ours: bytes, base: bytes, theirs: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        files = []
        for name, data in (("ours", ours), ("base", base), ("theirs", theirs)):
            files.append(Path(folder, name))
            files[-1].write_bytes(data)
        # The exit status counts conflicts; a negative status is an error.
        return git(
            "merge-file", "-p", f"--marker-size={MARKER}", *map(str, files),
            ok=tuple(range(128)),
        ).stdout


def clean_segments(merged: bytes) -> list[bytes]:
    """Split merge output into the text between its conflict regions."""
    segments, current, inside = [], [], False
    for line in merged.splitlines(keepends=True):
        if line.startswith(b"<" * MARKER + b" "):
            if inside:
                raise Mismatch("nested conflict markers")
            segments.append(b"".join(current))
            current, inside = [], True
        elif line.startswith(b">" * MARKER + b" "):
            inside = False
        elif not inside:
            current.append(line)
    if inside:
        raise Mismatch("unterminated conflict markers")
    segments.append(b"".join(current))
    return segments


def resolved_regions(path: str, segments: list[bytes], actual: bytes) -> list[str]:
    """Return the hand-resolved regions, or raise if anything else differs."""
    if len(segments) == 1:
        if actual != segments[0]:
            raise Mismatch(f"{path} does not equal the clean merge")
        return []
    if not actual.startswith(segments[0]):
        raise Mismatch(f"{path} differs from the merge before its first conflict")
    regions, position = [], len(segments[0])
    for index, segment in enumerate(segments[1:], start=1):
        last = index == len(segments) - 1
        found = len(actual) - len(segment) if last else actual.find(segment, position)
        if found < position or (last and not actual.endswith(segment)):
            raise Mismatch(f"{path} differs from the merge outside conflict {index}")
        first_line = actual.count(b"\n", 0, position) + 1
        text = actual[position:found].decode(errors="replace")
        regions.append(
            f"{path}:{first_line} (conflict {index})\n"
            + "".join(f"    | {line}\n" for line in text.splitlines())
        )
        position = found + len(segment)
    return regions


def verify(upstream: str, branches: list[str]) -> tuple[set[str], list[str]]:
    untracked = names("ls-files", "--others", "--exclude-standard", "--", TRAINER)
    changed = names("diff", "--name-only", upstream, "--", TRAINER) | untracked
    touched_by = {
        branch: names("diff", "--name-only", f"{upstream}...{branch}", "--", TRAINER)
        for branch in branches
    }
    stray = sorted(changed - set().union(*touched_by.values()))
    if stray:
        raise Mismatch(
            f"{TRAINER} differs from {upstream} in files no fix branch touches:\n  "
            + "\n  ".join(stray)
        )
    bases = {
        branch: git("merge-base", upstream, branch).stdout.decode().strip()
        for branch in branches
    }
    regions = []
    for path in sorted(set().union(*touched_by.values())):
        merged = content(upstream, path)
        for branch in branches:
            if path not in touched_by[branch]:
                continue
            theirs = content(branch, path)
            if theirs is None:
                raise Mismatch(f"{branch} deletes {path}; merge it by hand")
            base = content(bases[branch], path) or b""
            merged = merge_file(merged or b"", base, theirs)
        file = Path(path)
        actual = file.read_bytes() if file.is_file() else None
        if merged is None or actual is None:
            if merged != actual:
                raise Mismatch(f"{path} exists in only one of the tree and the merge")
            continue
        regions += resolved_regions(path, clean_segments(merged), actual)
    return changed, regions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", default=UPSTREAM)
    parser.add_argument(
        "--branch",
        dest="branches",
        action="append",
        help="fix branch to merge, in order (repeatable; default: the pr/* fixes)",
    )
    args = parser.parse_args()
    branches = args.branches or list(BRANCHES)
    try:
        changed, regions = verify(args.upstream, branches)
    except Mismatch as error:
        print(f"Trainer is not upstream plus fixes: {error}", file=sys.stderr)
        return 1
    print(
        f"{TRAINER} is {args.upstream} plus {len(branches)} fix branches: "
        f"{len(changed)} files differ from upstream and each matches its merge."
    )
    for path in sorted(changed):
        print(f"  {path}")
    if regions:
        print(f"{len(regions)} conflict regions are resolved by hand; review them:")
        print("".join(regions), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
