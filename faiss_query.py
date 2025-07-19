#!/usr/bin/env python3
import os, glob, json, numpy as np, faiss
from openai import OpenAI
from tqdm import tqdm

# Initialize client (uses OPENAI_API_KEY from environment)
client = OpenAI()
EMB = "text-embedding-3-small"

vectors = []
metas = []

for file in tqdm(glob.glob("data/*_biblio.json"), desc="Reading records"):
    with open(file) as f:
        records = json.load(f)

    for r in records:
        text = r.get("abstract") or r.get("title") or ""
        if not text.strip():
            continue

        # ✅ NEW API SYNTAX
        response = client.embeddings.create(model=EMB, input=text)
        vec = response.data[0].embedding

        vectors.append(vec)
        metas.append({
            "title": r.get("title", "No Title"),
            "pubno": r.get("pubno", "UNKNOWN"),
            "ipc": r.get("ipc", [])
        })

# Save FAISS index
vec_np = np.array(vectors, dtype="float32")
index = faiss.IndexFlatL2(vec_np.shape[1])
index.add(vec_np)
faiss.write_index(index, "patent.faiss")

# Save metadata
with open("patent_meta.json", "w") as f:
    json.dump(metas, f, indent=2)

print(f"✅ Done. Indexed {len(metas)} patents.")
"""
1. EP1000000  score=0.1588
   Title: Apparatus for manufacturing green bricks for the brick manufacturing industry
   IPC:   B28B5/02, B28B7/00, B28B1/29

2. EP1000000  score=0.1588
   Title: Apparatus for manufacturing green bricks for the brick manufacturing industry
   IPC:   B28B5/02, B28B7/00, B28B1/29
#!/usr/bin/env python3
import sys, json, numpy as np, faiss, os
from openai import OpenAI

EMB_MODEL = "text-embedding-3-small"
client = OpenAI()

if len(sys.argv) < 2:
    sys.exit("Usage: faiss_query.py <query text> [k]")

query_text = " ".join(sys.argv[1:-1]) if sys.argv[-1].isdigit() and len(sys.argv) > 2 else " ".join(sys.argv[1:])
k = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 5

# Load index + metadata
index = faiss.read_index("patent.faiss")
metas = json.load(open("patent_meta.json"))

# Embed query
resp = client.embeddings.create(model=EMB_MODEL, input=query_text)
qvec = np.array([resp.data[0].embedding], dtype="float32")
faiss.normalize_L2(qvec)

# Search
D, I = index.search(qvec, k)

print(f"\n🔎 Query: {query_text}\n")
for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
    if idx == -1:  # safety if fewer than k vectors
        continue
    m = metas[idx]
    print(f"{rank}. EP{m.get('pubno')}  score={dist:.4f}")
    print(f"   Title: {m.get('title')}")
    print(f"   IPC:   {', '.join(m.get('ipc', []))}\n")

"""