#!/usr/bin/env python3
"""
Parse an OPS “biblio” XML file into a compact JSON record.

Usage
-----
    ./scripts/parse_biblio.py data/EP1000000_biblio.xml
"""
import sys, json, xmltodict, pathlib

# ---- CLI guard --------------------------------------------------------------
if len(sys.argv) < 2:
    sys.exit("Usage: parse_biblio.py data/EPnnnnn_biblio.xml")

xml_path = pathlib.Path(sys.argv[1])

# ---- load & sanity‑check XML ------------------------------------------------
with xml_path.open() as f:
    data = xmltodict.parse(f.read())

if "error" in data:  # OPS sometimes wraps errors in XML
    sys.exit(f"❌ OPS error inside {xml_path.name}:\n{json.dumps(data['error'], indent=2)}")

wpd = data.get("ops:world-patent-data")
if not wpd:
    sys.exit("❌ <ops:world-patent-data> root not found")

# ---- locate <exchange-document> nodes regardless of wrapper -----------------
docs = (
    wpd.get("exchange-document")                                   # legacy
    or wpd.get("exchange-documents", {}).get("exchange-document")  # current
)
if not docs:
    sys.exit("❌ no <exchange-document> found – XML layout unknown")

if isinstance(docs, dict):   # normalise singleton ➜ list
    docs = [docs]

# ---- iterate & extract ------------------------------------------------------
records = []
for doc in docs:
    bib = doc.get("bibliographic-data")
    if not bib:
        continue  # skip malformed documents

    # Publication‐reference (pick first docdb id)
    pub_ids = bib["publication-reference"]["document-id"]
    pub_ids = pub_ids if isinstance(pub_ids, list) else [pub_ids]
    pub_id  = pub_ids[0]

    # Title – prefer English, else first found
    title_el = bib.get("invention-title", {})
    if isinstance(title_el, list):
        title_txt = next((t["#text"] for t in title_el
                          if t.get("@lang") == "en"), title_el[0]["#text"])
    elif title_el:
        title_txt = title_el["#text"]
    else:
        title_txt = ""

    # IPC codes (list of strings)
    ipc_block = bib.get("classification-ipc", {})
    if isinstance(ipc_block, dict):
        ipc_texts = ipc_block.get("text", [])
        ipc_list  = [ipc_texts] if isinstance(ipc_texts, str) else ipc_texts
    else:                       # already a list of <text>
        ipc_list = [c for c in ipc_block]

    records.append({
        "pubno": pub_id.get("doc-number", "").strip(),
        "date" : pub_id.get("date", "").strip(),
        "title": title_txt.strip(),
        "ipc"  : [c.strip() for c in ipc_list],
    })

# ---- write JSON -------------------------------------------------------------
out = xml_path.with_suffix(".json")
out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
print(f"📝 wrote {out} ({len(records)} record{'s' if len(records)!=1 else ''})")
