#!/usr/bin/env python3
import json, glob, pandas as pd
files = glob.glob("data/*_biblio.json")
records = []
for f in files:
    records += json.load(open(f))
pd.json_normalize(records).to_csv("patents.csv", index=False)
print("✅ patents.csv created:", len(records), "rows")
