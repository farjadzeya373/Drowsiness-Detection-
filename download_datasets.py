"""download_datasets.py
Attempts to download and extract a subset of the MRL Eye Dataset for training.
Because some research dataset sites require manual access or CAPTCHA, this script tries an automatic download and also provides manual instructions.
If automatic download fails, please download the dataset manually and place the images under dataset/eyes/open and dataset/eyes/closed.

USAGE:
    python download_datasets.py

Notes:
- Edit DATA_URL if MRL provides a direct download link for a zip file containing eye images.
- The script will attempt to extract a zip and organize images into dataset/eyes/open and dataset/eyes/closed.
"""

import os, sys, zipfile
from pathlib import Path
import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "dataset" / "eyes"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OPEN_DIR = DATA_DIR / "open"
CLOSED_DIR = DATA_DIR / "closed"
OPEN_DIR.mkdir(parents=True, exist_ok=True)
CLOSED_DIR.mkdir(parents=True, exist_ok=True)

# === IMPORTANT: set the actual MRL dataset download URL here if you have it ===
# Example placeholder (NOT an active URL). If MRL provides a direct zip link, replace below.
DATA_URL = "https://example.com/mrl_eye_dataset_subset.zip"

def download_file(url, dest):
    print(f"Downloading dataset from: {url}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print("Automatic download failed:", e)
        return False

def extract_zip(zip_path, extract_to):
    print("Extracting:", zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)

def organize_extracted(src_dir):
    """Look for folders or files matching 'open' and 'closed' and move them to dataset/eyes."""
    for root, dirs, files in os.walk(src_dir):
        for d in dirs:
            if 'open' in d.lower():
                src = Path(root) / d
                for f in src.glob('*.*'):
                    f.rename(OPEN_DIR / f.name)
            if 'closed' in d.lower():
                src = Path(root) / d
                for f in src.glob('*.*'):
                    f.rename(CLOSED_DIR / f.name)

def main():
    print("MRL Eye Dataset automatic downloader (attempt)")
    if DATA_URL.startswith('https://example.com'):
        print("NOTE: DATA_URL is a placeholder. If you have a direct MRL dataset zip link, edit DATA_URL in this file and re-run.")
        print("Falling back to manual instructions...") 
        print("\nManual steps:\n1) Visit the MRL Eye Dataset page, download the dataset zip to your machine.\n2) Extract contents and copy open-eye images to dataset/eyes/open and closed-eye images to dataset/eyes/closed.\n3) Run train_eye_model.py to train the model.")
        return

    zip_path = BASE_DIR / "mrl_eye_subset.zip"
    ok = download_file(DATA_URL, zip_path)
    if not ok:
        print("Download failed. See README for manual download instructions.")
        return
    extract_dir = BASE_DIR / "mrl_extracted"
    extract_dir.mkdir(exist_ok=True)
    extract_zip(zip_path, extract_dir)
    organize_extracted(extract_dir)
    print("Dataset organized under dataset/eyes/open and dataset/eyes/closed")


if __name__ == '__main__':
    main()
