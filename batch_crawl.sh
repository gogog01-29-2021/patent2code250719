#!/usr/bin/env bash
set -euo pipefail
LIST=${1:-patents.txt}
while read pn; do
  ./scripts/fetch_biblio.sh "$pn" || echo "fail $pn" >> crawl_errors.log
done < "$LIST"
