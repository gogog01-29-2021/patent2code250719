#!/usr/bin/env bash
# Fetch / refresh EPO‑OPS token → .token & .token_exp
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/.env"                    # EPO_CONSUMER_KEY, *_SECRET, TOKEN_TTL (opt)

: "${EPO_CONSUMER_KEY:?}"; : "${EPO_CONSUMER_SECRET:?}"
TOKEN_TTL="${TOKEN_TTL:-3500}"         # default 58 min

AUTH=$(printf "%s:%s" "$EPO_CONSUMER_KEY" "$EPO_CONSUMER_SECRET" | base64 -w0)

RAW=$(curl -s -X POST "https://ops.epo.org/3.2/auth/accesstoken" \
  -H "Authorization: Basic $AUTH" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H "Accept: application/json" \
  -d "grant_type=client_credentials")

echo "$RAW" > /tmp/ops_token_resp.json   # for debug

TOKEN=$(echo "$RAW" | jq -r '.access_token // empty')
if [[ -z "$TOKEN" ]]; then
  echo "❌ OAuth error – check /tmp/ops_token_resp.json"
  exit 1
fi

echo "$TOKEN"                           >  "$ROOT/.token"
echo $(( $(date +%s) + TOKEN_TTL ))     >  "$ROOT/.token_exp"
echo "✅ new token saved to .token (valid ~$(($TOKEN_TTL/60)) min)"
