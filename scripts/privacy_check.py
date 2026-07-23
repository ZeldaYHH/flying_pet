from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "macOS user path": re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    "local hostname": re.compile(rb"\b[A-Za-z0-9._-]+\.local\b"),
    "credential assignment": re.compile(
        rb"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd)"
        rb"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{8,}"
    ),
    "authorization token": re.compile(
        rb"(?i)\b(authorization:\s*(bearer|basic)|bearer\s+)"
        rb"[A-Za-z0-9_./+=-]{8,}"
    ),
}
EMAIL_PATTERN = re.compile(
    rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
ALLOWED_EMAIL_SUFFIX = b"@users.noreply.github.com"


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def tracked_files() -> list[Path]:
    output = git("ls-files", "-z").stdout
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def scan_bytes(label: str, content: bytes) -> list[str]:
    findings = []
    for name, pattern in TEXT_PATTERNS.items():
        if pattern.search(content):
            findings.append(f"{label}: {name}")
    for match in EMAIL_PATTERN.finditer(content):
        email = match.group(0)
        if not email.lower().endswith(ALLOWED_EMAIL_SUFFIX):
            findings.append(f"{label}: non-noreply email")
    return findings


def scan_tracked_content() -> list[str]:
    findings = []
    for path in tracked_files():
        if not path.is_file():
            continue
        try:
            content = path.read_bytes()
        except OSError as exc:
            findings.append(f"{path.relative_to(ROOT)}: cannot read ({exc})")
            continue
        findings.extend(scan_bytes(str(path.relative_to(ROOT)), content))
    return findings


def scan_commit_identities() -> list[str]:
    result = git(
        "log",
        "--all",
        "--format=%H%x00%ae%x00%ce",
        check=False,
    )
    findings = []
    for line in result.stdout.splitlines():
        parts = line.split(b"\0")
        if len(parts) != 3:
            continue
        commit, author_email, committer_email = parts
        for role, email in (
            ("author", author_email),
            ("committer", committer_email),
        ):
            if email and not email.lower().endswith(ALLOWED_EMAIL_SUFFIX):
                findings.append(
                    f"commit {commit.decode()[:12]}: {role} uses non-noreply email"
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reject tracked secrets and machine-identifying metadata."
    )
    parser.add_argument(
        "--pre-push",
        action="store_true",
        help="Run in a Git pre-push hook.",
    )
    parser.parse_args()

    findings = scan_tracked_content() + scan_commit_identities()
    if findings:
        print("Privacy check failed:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        print("Push blocked until these findings are sanitized.", file=sys.stderr)
        return 1

    print("Privacy check passed: no tracked secrets or machine identifiers found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
