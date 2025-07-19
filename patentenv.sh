#!/usr/bin/env bash
set -euo pipefail                     # safe‑bash boilerplate

# ---- 0. prerequisites (jq, curl, base64 come with coreutils) ----
# mac:  brew install jq
# deb:  sudo apt-get install -y jq coreutils curl

# ---- 1. secrets (rotate these after pasting them here!) ----
CONSUMER_KEY="IAlIF82RV7dS7P9eMOyB8igK9BuqxhHsYGYoRrgmXmEBDZ25"
CONSUMER_SECRET="7GCSocco9cGp1u9LpPJq5IpZvGWtaHrGHtSahAVnZLjuJAFF8AHxAOCo59C2VIXt"

# ---- 2. build Basic‑Auth header ----
AUTH=$(printf "%s:%s" "$CONSUMER_KEY" "$CONSUMER_SECRET" | base64 -w0)

# ---- 3. grab the token ----
# ---- 3. grab the token ----
TOKEN=$(curl -s -X POST "https://ops.epo.org/3.2/auth/accesstoken" \
  -H "Authorization: Basic $AUTH" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H "Accept: application/json" \
  -d "grant_type=client_credentials" | \
  tee /tmp/ops_token_resp.json | jq -r .access_token)


echo "✅ Got token: ${TOKEN:0:20}…"

# ---- 4. hit an endpoint ----
curl -s -H "Authorization: Bearer $TOKEN" \
       -H "Accept: application/xml" \
       "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/EP1000000/biblio" \
  | head -20
