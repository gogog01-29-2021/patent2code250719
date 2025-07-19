#!/usr/bin/env bash
# Usage: fetch_biblio.sh EP1000000
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/.env"
PUB=${1:? "Usage: fetch_biblio.sh EP1000000"}

# refresh if token missing or expired
if [[ ! -f "$ROOT/.token_exp" || $(<"$ROOT/.token_exp") -lt $(date +%s) ]]; then
  "$ROOT/scripts/get_token.sh"
fi
TOKEN=$(<"$ROOT/.token")

OUT="$ROOT/data/${PUB}_biblio.xml"
mkdir -p "$ROOT/data"

STATUS=$(curl -s -w '%{http_code}' -o "$OUT" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/xml" \
  "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/${PUB}/biblio")

if [[ "$STATUS" != "200" ]]; then
  rm -f "$OUT"
  echo "❌ OPS returned HTTP $STATUS – likely bad token or quota. Check /tmp/ops_biblio_${PUB}.xml"
  exit 1
fi

echo "📥 saved → $OUT"
