#!/usr/bin/env bash
# Install these pointer notes as Claude project memory on whatever machine you are on.
# The memory directory is keyed by the project's ABSOLUTE path, so it differs per machine.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
slug="$(echo "$repo" | sed 's|/|-|g')"
dest="$HOME/.claude/projects/${slug}/memory"
mkdir -p "$dest"
cp "$(dirname "${BASH_SOURCE[0]}")"/*.md "$dest/"
echo "installed into $dest"
ls "$dest"
