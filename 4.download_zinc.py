# Original code provided by Prof. Suvamay Jana, CAD course, IIT Dharwad.
# Shared with permission. Modified by Barshan.

import os
import requests
import time

URI_FILE = "ZINC-downloader-2D-smi.uri"
OUTPUT_DIR = "zinc_smi_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(URI_FILE) as f:
    urls = [line.strip() for line in f if line.strip()]

print(f"Downloading {len(urls)} files to '{OUTPUT_DIR}/'...")

for url in urls:
    filename = url.split("/")[-1]
    out_path = os.path.join(OUTPUT_DIR, filename)

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        with open(out_path, "w") as f:
            f.write(r.text)
        print(f"  ✓ {filename} ({len(r.text)} chars)")
    except Exception as e:
        print(f"  ✗ {filename}: {e}")
    time.sleep(0.5)

print("Done.")
