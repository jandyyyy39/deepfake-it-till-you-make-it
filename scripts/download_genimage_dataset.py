"""
Deepfake It Till You Make It: Standardized GenImage Downloader
===========================================================
Standardized downloader for the group project. 

OUTPUT DIRECTORY:
    Automatically saves to: ./datasets/genimage/<model_name>/

USAGE:
    python download_genimage.py --model sdv5
    python download_genimage.py --model midjourney

NOTES:
    - Automatically skips existing files.
    - Bypasses Google Drive's large-file virus scan warnings.
    - Forces correct naming conventions for split-zip concatenation.
"""
import re
import json
import argparse
import gdown
from pathlib import Path

def get_id_from_url(url):
    """Extracts the unique Google Drive File ID."""
    patterns = [r'd/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

def main():
    parser = argparse.ArgumentParser(description="Deepfake It: GenImage Downloader")
    parser.add_argument("--model", type=str, required=True, 
                        choices=['adm', 'biggan', 'glide', 'midjourney', 'sdv4', 'sdv5', 'vqdm', 'wukong'],
                        help="Generator model split to download")
    args = parser.parse_args()

    # Load Data
    try:
        with open('./scripts/genimage_links.json', 'r') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print("[ERROR] 'genimage_links.json' not found. Please ensure it's in the same folder.")
        return

    # Hardcoded Pathing Logic
    # Anchors to <project_root>/datasets/genimage/<model>/
    DOWNLOAD_DIR = (Path(__file__).parent / "../datasets/genimage" / args.model).resolve()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    model_links = full_data[args.model]
    
    print(f"\n" + "="*50)
    print(f" PROJECT: Deepfake It Till You Make It")
    print(f" MODEL  : {args.model.upper()}")
    print(f" TARGET : {DOWNLOAD_DIR}")
    print("="*50 + "\n")

    # Download
    for filename, url in model_links.items():
        file_id = get_id_from_url(url)
        output_path = os.path.join(DOWNLOAD_DIR, filename)

        if os.path.exists(output_path):
            print(f"[SKIP] {filename} (Already exists)")
            continue

        print(f"[START] {filename}")
        try:
            # High-level gdown call handles binary stream and security bypasses
            gdown.download(id=file_id, output=output_path, quiet=False)
        except Exception as e:
            print(f"[FAILED] {filename}: {e}")

    print(f"\n[DONE] All {args.model} parts processed.")
    print(f"Next step: cd {DOWNLOAD_DIR} && cat {args.model}.z* {args.model}.zip > combined.zip")

if __name__ == "__main__":
    main()