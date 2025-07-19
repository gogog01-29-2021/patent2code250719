#!/usr/bin/env bash
set -euo pipefail
find data -name '*_biblio.xml' -print0 | xargs -0 -n1 ./scripts/parse_biblio.py
