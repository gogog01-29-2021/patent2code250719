#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUB=${1:?Usage: fetch_fulltext.sh EP1000000}
source "$ROOT/.env"

# refresh token if needed
if [[ ! -f "$ROOT/.token_exp" || $(<"$ROOT/.token_exp") -lt $(date +%s) ]]; then
  "$ROOT/scripts/get_token.sh"
fi
TOKEN=$(<"$ROOT/.token")

OUT="$ROOT/data/${PUB}_abstract.xml"
curl -s -H "Authorization: Bearer $TOKEN" \
     -H "Accept: application/xml" \
     "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/${PUB}/fulltext/abstract" \
     -o "$OUT"
echo "📥 abstract → $OUT"
