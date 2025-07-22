#!/usr/bin/env python3
import argparse, subprocess, sys, pathlib, json, glob

ROOT = pathlib.Path(__file__).parent

def run(cmd):
    print(f"▶ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        sys.exit(r.returncode)

def crawl(pubs_file):
    for pn in open(pubs_file):
        pn = pn.strip()
        if not pn: continue
        run(f"./scripts/fetch_biblio.sh {pn}")
        xml = ROOT / f"data/{pn}_biblio.xml"
        if xml.exists():
            run(f"./scripts/parse_biblio.py {xml}")
    summarize()

def summarize():
    xmls = glob.glob("data/*_biblio.xml")
    jsons = glob.glob("data/*_biblio.json")
    print(f"📊 XML={len(xmls)} JSON={len(jsons)}")

def build_index():
    run("python3 faiss_build.py")
    run("python3 - <<'PY'\nimport faiss,json; i=faiss.read_index('patent.faiss'); m=json.load(open('patent_meta.json')); print(f'Index {i.ntotal} vectors; meta {len(m)} records')\nPY")

def query(q):
    run(f"./faiss_query.py \"{q}\"")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["crawl","index","query","summary"])
    ap.add_argument("--pubs")
    ap.add_argument("--q")
    args = ap.parse_args()

    if args.action=="crawl": crawl(args.pubs or "patents.txt")
    elif args.action=="index": build_index()
    elif args.action=="query": query(args.q or "concrete mixer")
    elif args.action=="summary": summarize()
"""
python3 main.py crawl --pubs patents.txt
python3 main.py index
python3 main.py query --q "extrusion brick mold"
python3 main.py summary

"""