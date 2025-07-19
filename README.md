# patent2code
# patent2code

**Mission:** Turn raw patent numbers → structured domain knowledge → semantic index → LLM‑assisted design specs → (eventually) parametric CAD + iterative 3D refinement.

This README is an *execution playbook*: every phase has (1) goal, (2) concrete commands, (3) artifacts, (4) why it matters for the next hop.

---

## High‑Level Pipeline

| Phase | Goal                         | Run Commands (from project root)                                                                                           | Output                              | Bridge to next                 |
| ----- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------ |
| 1     | Token + single biblio sanity | `./scripts/get_token.sh`<br>`./scripts/fetch_biblio.sh EP1000000`<br>`./scripts/parse_biblio.py data/EP1000000_biblio.xml` | JSON record                         | Proves pipeline wiring         |
| 2     | Bulk crawl metadata          | `./batch_crawl.sh patents.txt`                                                                                             | XML files                           | Raw corpus                     |
| 3     | Bulk parse JSON              | `./bulk_parse.sh`                                                                                                          | `*_biblio.json`                     | Structured fields (title, IPC) |
| 4     | Merge & explore              | `./merge_to_csv.py`                                                                                                        | `patents.csv`                       | Inspect dataset; filter domain |
| 5     | Abstracts & claims text      | `while read p; do ./scripts/fetch_fulltext.sh "$p"; done < patents.txt`                                                    | `*_abstract.xml`                    | Natural language for NLP       |
| 6     | Embed & index                | `python faiss_build.py`                                                                                                    | `patent.faiss` + `patent_meta.json` | Fast semantic retrieval        |
| 7     | Query                        | `./faiss_query.py "self-healing concrete"`                                                                                 | Ranked hits                         | Pull candidate patents         |
| 8     | Patent → design spec (NLP)   | *(future)* `spec_extractor.py` → load abstract+claims → LLM → components, dimensions, materials schema.                    | JSON specs                          | Input to CAD synthesis         |
| 9     | Spec → parametric CAD        | *(future)* OpenSCAD / FreeCAD scripts generate `.scad` / `.step`.                                                          | 3D model files                      | Visualization, simulation      |
| 10    | 3D reconstruction refinement | *(future)* Geometry heuristics + 3D diffusion / shape libs refine & iterate.                                               | Improved models                     | Prototype / simulation         |

---

## Repository Layout (Target)

```
patent2code/
  scripts/
    get_token.sh            # Obtain OPS token & cache expiry
    fetch_biblio.sh         # Fetch single bibliographic XML
    fetch_fulltext.sh       # Fetch abstract / claims XML
    parse_biblio.py         # Parse one XML → JSON
  batch_crawl.sh            # Loop over patent list → raw XML
  bulk_parse.sh             # Loop XML → JSON
  merge_to_csv.py           # Aggregate JSON → CSV
  faiss_build.py            # Build embedding + FAISS index (batch, cost logging)
  faiss_query.py            # Query index
  spec_extractor.py         # (future) LLM prompt → structured spec
  cad/
    generate_scad.py        # (future) Parametric CAD emitter
  data/
    raw/                    # Original XML (kept, but .gitignore large?)
    json/                   # Parsed JSON records
    embeds/                 # FAISS + metadata
  models/                   # (optional future) local models / adapters
  notebooks/                # EDA / prototyping
  config/
    embedding.yaml          # Provider, model, batch size, rate limits
  tests/
  .env.example
  .gitignore
  Makefile
  README.md
```

> **Note:** Large raw XML, embeddings, models should be excluded via `.gitignore`; keep *manifests* (lists, hashes) under version control.

---

## Setup

1. **Clone:**

   ```bash
   git clone git@github.com:gogog01-29-2021/patent2code250719.git
   cd patent2code250719
   ```
2. **Environment:** (example Conda; you can also use `uv` / `pip`)

   ```bash
   conda create -n patent2code python=3.11 -y
   conda activate patent2code
   pip install -r requirements.txt
   ```
3. **Configure secrets:**

   ```bash
   cp .env.example .env
   # edit OPENAI_API_KEY=... (or other provider keys)
   ```
4. **OPS (EPO) credentials:** Some endpoints are free but rate‑limited; if you have specific keys add them to `.env` (never commit .env).

---

## Minimal First Run (Phase 1)

```bash
./scripts/get_token.sh
./scripts/fetch_biblio.sh EP1000000
./scripts/parse_biblio.py data/EP1000000_biblio.xml > data/json/EP1000000_biblio.json
cat data/json/EP1000000_biblio.json | jq '.title, .applicants'
```

---

## Batch Workflow Cheatsheet

| Task            | Command                                                                 | Notes                                                  |
| --------------- | ----------------------------------------------------------------------- | ------------------------------------------------------ |
| Crawl list      | `./batch_crawl.sh patents.txt`                                          | Expects one patent number per line (EP, US, WO etc.)   |
| Parse all       | `./bulk_parse.sh`                                                       | Writes JSON to `data/json/`                            |
| Merge CSV       | `./merge_to_csv.py -o patents.csv`                                      | Creates tabular dataset (title, IPC, date, applicants) |
| Fetch abstracts | `while read p; do ./scripts/fetch_fulltext.sh "$p"; done < patents.txt` | Extend for claims if needed                            |
| Build index     | `python faiss_build.py --input patents.csv --out embeds/`               | Produces `patent.faiss`, `patent_meta.json`            |
| Query           | `./faiss_query.py "keyword phrase"`                                     | Returns top‑k JSON with scores                         |

---

## Data Schema (Parsed JSON) — Draft

```json
{
  "doc_id": "EP1000000",
  "publication_date": "2000-05-24",
  "application_date": "1998-11-13",
  "ipc": ["C04B14/10", "C04B28/02"],
  "title": "Self-healing cementitious composition",
  "applicants": ["Example Corp"],
  "inventors": ["Doe, Jane"],
  "abstract_text": "...",             // populated after fulltext fetch
  "claims_text": "...",               // (future) extended fetch
  "language": "EN",
  "source_xml": "data/raw/EP1000000_biblio.xml"
}
```

---

## Embedding & Indexing

**faiss\_build.py** (spec plan):

* Streams JSON/CSV.
* Chunks text (title + abstract) with sliding window (e.g. 512 token target) → embedding calls.
* Batch size + rate limiting; cost logging output `cost_log.jsonl` with fields:

  ```json
  {"ts": "2025-07-19T09:41:00Z", "model": "text-embedding-3-large", "n_tokens": 8732, "est_cost_usd": 0.0043}
  ```
* Builds FAISS index (L2 or inner product). Saves mapping id → `doc_id`, offset.

Query script merges top‑k hits with metadata, optionally re‑ranks with a cross‑encoder (future).

---

## Planned NLP Spec Extraction (Phase 8)

**Goal:** Convert abstract + earliest independent claim into structured schema:

```json
{
  "doc_id": "EP1000000",
  "domain": "construction",
  "components": ["binder", "microcapsule additive", "fiber"],
  "materials": {"binder": "Portland cement", "microcapsule": "polymer shell"},
  "dimensions": {"capsule_diameter_um": {"value": 110, "range": [80,140]}},
  "functions": ["self-healing", "crack sealing"],
  "constraints": ["compressive_strength >= 40 MPa"]
}
```

Prompt template + validation (JSON schema) + retry on malformed.

---

## Parametric CAD (Phase 9) — Roadmap

* Map `dimensions` + component taxonomy → OpenSCAD modules.
* Generate `.scad` + preview via CLI.
* Export `.stl` / `.step` for simulation.
* Track version lineage: patent\_doc\_id → spec\_hash → cad\_hash.

---

## 3D Iterative Refinement (Phase 10)

* Geometry metrics (volume, surface area, aspect ratios) fed back to LLM for constraint satisfaction.
* Optional 3D diffusion (e.g. point‑e / Shap‑E style) for aesthetic / ergonomic optimization.

---

## Makefile Targets (Suggested)

```
make env           # create & sync environment
make crawl         # batch_crawl.sh patents.txt
make parse         # bulk_parse.sh
make csv           # merge_to_csv.py
make abstracts     # fetch abstracts loop
make embed         # build FAISS index
make query Q="self-healing concrete"
make clean_raw     # rm -rf data/raw/*.xml
make clean_index   # rm embeds/patent.faiss embeds/patent_meta.json
```

---

## Security & Hygiene

| Risk               | Mitigation                                                                 |
| ------------------ | -------------------------------------------------------------------------- |
| API key leakage    | `.env` in `.gitignore`; template `.env.example`; pre-commit secret scan    |
| Oversized binaries | Keep installers & large datasets out (`data/raw` optionally ignored)       |
| Repro failures     | Capture versions: `requirements.txt` / lock file, embed model name in meta |
| Ambiguous parsing  | Unit tests on sample XML; checksum raw files                               |

Pre-commit hook example (local):

```bash
#!/usr/bin/env bash
if git diff --cached -U0 | grep -E 'sk-[A-Za-z0-9]{10,}' >/dev/null; then
  echo 'Secret-like pattern (sk-) found. Commit blocked.'
  exit 1
fi
```

---

## Testing Strategy

| Layer                   | Tooling                | Example                                                     |
| ----------------------- | ---------------------- | ----------------------------------------------------------- |
| XML parser              | `pytest`               | Given sample XML, assert IPC list parsed correctly          |
| Merge script            | `pytest`               | Synthetic 2 JSON inputs → one CSV row with combined IPC set |
| Embedding builder       | Mock provider          | Inject fake embeddings; assert FAISS dim & count            |
| Query                   | Integration test       | Small corpus (5 docs) known nearest neighbor                |
| Spec extractor (future) | JSON schema validation | Prompt returns valid, normalized units                      |

---

## Metrics & Logging

| Metric                       | Source                       | Purpose            |
| ---------------------------- | ---------------------------- | ------------------ |
| # patents parsed             | Counter in `merge_to_csv.py` | Coverage tracking  |
| Avg tokens / embedding       | Embedding log                | Cost estimation    |
| Embedding cost USD           | `cost_log.jsonl` aggregator  | Budget control     |
| Query latency                | `faiss_query.py` timing      | Performance tuning |
| Spec extraction success rate | JSON schema validator        | Reliability        |

---

## Roadmap (Condensed)

1. ✅ Single patent end‑to‑end sanity.
2. 🔄 Scale crawl + parsing (parallelization, retry logic, backoff).
3. 🔄 Embedding pipeline: streaming, incremental index updates.
4. 🟡 Spec extraction (LLM prompting + schema validation).
5. 🟡 Parametric CAD generation.
6. 🟡 Iterative refinement loop (constraints + geometry analysis).
7. 🟡 Web UI / API (serve search + spec + model preview).

Legend: ✅ done / 🔄 in progress / 🟡 planned.

---

## Contribution Guidelines (Early Stage)

* Open issue before large refactors.
* Use Conventional Commits (`feat:`, `chore:`, `fix:`, `refactor:`).
* Prefer small PRs (<400 LOC diff) with test coverage for parsing & index changes.
* Document new scripts in the phase table.

---

## License

TBD (MIT? Apache-2.0?). Add before first public release.

---

## Quick FAQ

**Q: Why not store raw claims now?**  Rate limits + bandwidth; start small (abstract+title) → expand.

**Q: Why FAISS over pure DB full‑text?**  Semantic nearest‑neighbor for concept similarity (materials function, cross‑lingual) beats lexical matching alone.

**Q: How to update index incrementally?**  Append new embeddings + add IDs to FAISS; preserve `patent_meta.json` (monotonic ID assignment), or rebuild nightly when corpus growth > X%.

---

## Next Steps for You

1. Populate `patents.txt` with a tight domain subset (e.g., self‑healing materials, additive manufacturing).
2. Run Phases 1–4, verify CSV quality (dates, IPC distribution).
3. Instrument embedding cost; set a daily token budget.
4. Draft the first `spec_extractor.py` prompt (we can co‑design it next).

*Let’s build the spec extractor + cost‑aware embed module next.*
