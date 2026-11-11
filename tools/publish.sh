#!/bin/bash
# Publish the current HEAD tree to the public repository as one snapshot commit.
# The working history stays local; the scan below refuses to publish private names.
set -euo pipefail
cd "$(dirname "$0")/.."
PUB="${WW_PUBLIC_DIR:-$HOME/.cache/writewell-public}"
DENY="${WW_DENY_FILE:-$HOME/.config/writewell/deny.txt}"       # one regex per line, kept outside the repo
[ -f "$DENY" ] || { echo "missing deny list: $DENY"; exit 1; }
if git grep -nIiE -f "$DENY" HEAD -- . ; then echo "refusing to publish: private names found"; exit 1; fi
if git grep -nIE "(sk-ant-|sk-or-|ghp_|sb_secret_|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)" HEAD -- . ":(exclude)tools/publish.sh" ; then echo "refusing: secret-like string"; exit 1; fi
[ -d "$PUB/.git" ] || { echo "public clone not found at $PUB"; exit 1; }
find "$PUB" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git archive HEAD | tar -x -C "$PUB"
cd "$PUB" && git add -A
git diff --cached --quiet && { echo "nothing to publish"; exit 0; }
git commit -qm "${1:-snapshot $(date -u +%Y-%m-%dT%H:%MZ)}"
git push -q origin HEAD:main && echo "published $(git rev-parse --short HEAD)"
