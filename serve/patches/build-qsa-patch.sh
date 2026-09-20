#!/usr/bin/env bash
# Compatibility entry point for older quick-start instructions.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo 'The old QSA-only patch is superseded. Building the complete 0.5.20 runtime.'
exec "$HERE/../build-image.sh"
