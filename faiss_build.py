#!/usr/bin/env python3
import os, glob, json, math, numpy as np, faiss
from openai import OpenAI
from tqdm import tqdm   # optional: pip install tqdm

EMB_MODEL = "text-embedding-3-small"
client = OpenAI()

records = []
for path in glob.glob("data/*_biblio.json"):
    with open(path) as f:
        records.extend(json.load(f))

# Prepare texts
texts = [f"{r.get('title','')} IPC {', '.join(r.get('ipc', []))}" for r in records]

BATCH = 100  # OpenAI embeddings endpoint allows batching; tweak as needed
vectors = []

for i in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
    batch = texts[i:i+BATCH]
    resp = client.embeddings.create(model=EMB_MODEL, input=batch)
    vectors.extend([d.embedding for d in resp.data])

vec_np = np.array(vectors, dtype="float32")
index = faiss.IndexFlatIP(vec_np.shape[1])
# Normalize for cosine similarity style (optional)
faiss.normalize_L2(vec_np)
index.add(vec_np)
faiss.write_index(index, "patent.faiss")
with open("patent_meta.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"✅ Indexed {len(records)} docs into patent.faiss (dim={vec_np.shape[1]})")
