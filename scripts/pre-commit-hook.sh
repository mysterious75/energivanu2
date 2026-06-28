#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Git Pre-Commit Hook — NC-Licence Compliance Check
# ===================================================
# Runs check_compliance.py on staged files only.
# Blocks the commit if NC-licensed data contamination is found.
#
# INSTALLATION:
#   Option A — Symlink (recommended):
#     ln -sf ../../scripts/pre-commit-hook.sh .git/hooks/pre-commit
#
#   Option B — Copy:
#     cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
#     chmod +x .git/hooks/pre-commit
#
#   Option C — Via pre-commit framework:
#     Add to .pre-commit-config.yaml:
#       repos:
#         - repo: local
#           hooks:
#             - id: nc-compliance
#               name: NC-Licence Compliance
#               entry: scripts/pre-commit-hook.sh
#               language: script
#               pass_filenames: false
#
# REQUIREMENTS:
#   - Python 3.10+
#   - pyyaml (pip install pyyaml)
#
# To skip this hook for a single commit:
#   git commit --no-verify
# ---------------------------------------------------------------------------

set -euo pipefail

# ── Colour codes ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Locate repo root ──────────────────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
    echo -e "${RED}ERROR: Not inside a git repository.${RESET}" >&2
    exit 1
fi

SCANNER="$REPO_ROOT/scripts/check_compliance.py"
if [[ ! -f "$SCANNER" ]]; then
    echo -e "${YELLOW}WARNING: check_compliance.py not found at $SCANNER — skipping.${RESET}"
    exit 0
fi

# ── Run scanner on staged files ───────────────────────────────────────
echo -e "${CYAN}${BOLD}Running NC-licence compliance check...${RESET}"

OUTPUT_FILE="$(mktemp)"
trap 'rm -f "$OUTPUT_FILE"' EXIT

if python3 "$SCANNER" --root "$REPO_ROOT" --staged --format text > "$OUTPUT_FILE" 2>&1; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi

# ── Display results ───────────────────────────────────────────────────
cat "$OUTPUT_FILE"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}✅ Compliance check PASSED — commit allowed.${RESET}"
elif [[ $EXIT_CODE -eq 1 ]]; then
    echo ""
    echo -e "${RED}${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${RED}${BOLD}║  ❌  COMMIT BLOCKED — NC-licensed data contamination found  ║${RESET}"
    echo -e "${RED}${BOLD}╠══════════════════════════════════════════════════════════════╣${RESET}"
    echo -e "${RED}${BOLD}║  Remove NC data/files before committing.                    ║${RESET}"
    echo -e "${RED}${BOLD}║  See data/README.md for licence details.                    ║${RESET}"
    echo -e "${RED}${BOLD}║                                                             ║${RESET}"
    echo -e "${RED}${BOLD}║  To skip this check (NOT recommended):                      ║${RESET}"
    echo -e "${RED}${BOLD}║    git commit --no-verify                                   ║${RESET}"
    echo -e "${RED}${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
else
    echo -e "${YELLOW}${BOLD}⚠️  Scanner error (exit code $EXIT_CODE) — allowing commit.${RESET}"
    exit 0
fi

exit $EXIT_CODE
