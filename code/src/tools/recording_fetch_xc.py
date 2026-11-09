"""Download Common Snipe (Gallinago gallinago) recordings, quality A and B,
from Xeno-canto (API v3) and save a metadata CSV alongside the audio.

Usage:
    export XC_API_KEY="your_key"      # from your xeno-canto account page
    python download_xc_snipe.py
"""
import csv
import os
import time

import requests

API_URL = "https://xeno-canto.org/api/3/recordings"
API_KEY = os.environ["XC_API_KEY"]
QUALITIES = ["A", "B"]
OUT_DIR = "./code/data/xc_gallinago_gallinago"
PER_PAGE = 500          # API maximum
PAUSE = 1.0             # seconds between requests, be polite to the server

FIELDS = ["id", "gen", "sp", "en", "type", "sex", "stage", "q", "length",
          "rec", "cnt", "loc", "lat", "lon", "date", "time", "lic", "url",
          "file-name", "rmk"]


def get_json(params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  request failed ({e}), retry {attempt + 1}/{retries}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("API request failed repeatedly")


def fetch_metadata(quality):
    """Return all recording records for one quality rating."""
    query = f'gen:gallinago sp:gallinago q:{quality}'
    recordings, page = [], 1
    while True:
        data = get_json({"query": query, "key": API_KEY,
                         "page": page, "per_page": PER_PAGE})
        if "error" in data:
            raise RuntimeError(f"API error: {data}")
        recordings += data["recordings"]
        print(f"Quality {quality}: page {page}/{data['numPages']} "
              f"({len(recordings)}/{data['numRecordings']})")
        if page >= int(data["numPages"]):
            return recordings
        page += 1
        time.sleep(PAUSE)


def download(rec):
    name = rec.get("file-name") or f"XC{rec['id']}.mp3"
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skipped"
    url = rec["file"]
    if url.startswith("//"):
        url = "https:" + url
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            os.replace(path + ".part", path)
            return "ok"
        except requests.RequestException as e:
            print(f"  download failed for XC{rec['id']} ({e})")
            time.sleep(5 * (attempt + 1))
    return "failed"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_recs = []
    for q in QUALITIES:
        all_recs += fetch_metadata(q)
        time.sleep(PAUSE)

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_recs)
    print(f"Saved metadata for {len(all_recs)} recordings.")

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for i, rec in enumerate(all_recs, 1):
        status = download(rec)
        counts[status] += 1
        print(f"[{i}/{len(all_recs)}] XC{rec['id']} {status}")
        if status == "ok":
            time.sleep(PAUSE)
    print("Done:", counts)


if __name__ == "__main__":
    main()