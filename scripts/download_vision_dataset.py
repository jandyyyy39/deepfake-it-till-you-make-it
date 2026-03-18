"""
Script to download images from the VISION dataset
Downloads only the first 10 images from the 'flat' subdirectory of each camera type
"""

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
import time
from pathlib import Path

# Configuration
BASE_URL = "https://lesc.dinfo.unifi.it/VISION/dataset/"
DOWNLOAD_DIR = (Path(__file__).parent / "../datasets/vision").resolve()
DELAY_BETWEEN_REQUESTS = 0.5  # seconds, to be polite to the server

def get_directory_listing(url):
    """Parse an Apache directory listing page and return list of items"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links in the directory listing
        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and not href.startswith('?') and href != '../':
                links.append(href)
        
        return links
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def is_image_file(filename):
    """Check if a file is an image based on extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}
    return Path(filename).suffix.lower() in image_extensions

def download_file(url, save_path):
    """Download a file from URL to save_path"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(save_path):
            print(f"  Already exists: {os.path.basename(save_path)}")
            return True
        
        # Download the file
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        # Write to file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  DOWNLOADED: {os.path.basename(save_path)}")
        return True
        
    except Exception as e:
        print(f"  ERROR DOWNLOADING {url}: {e}")
        return False

def main():
    print("=" * 70)
    print("VISION Dataset Downloader")
    print("Downloading images from 'flat' subdirectories only")
    print("=" * 70)
    
    # Get list of camera directories
    print(f"\nFetching camera directories from {BASE_URL}...")
    camera_dirs = get_directory_listing(BASE_URL)
    
    # Filter for directories (they end with /)
    camera_dirs = [d for d in camera_dirs if d.endswith('/')]
    print(f"Found {len(camera_dirs)} camera directories")
    
    # Statistics
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    
    # Process each camera directory
    for i, camera_dir in enumerate(camera_dirs, 1):
        camera_name = camera_dir.rstrip('/')
        print(f"\n[{i}/{len(camera_dirs)}] Processing: {camera_name}")
        
        # Construct URL to images/flat/ directory
        flat_url = urljoin(BASE_URL, f"{camera_dir}images/flat/")
        print(f"  URL: {flat_url}")
        
        # Get list of images in flat directory
        time.sleep(DELAY_BETWEEN_REQUESTS)
        image_files = get_directory_listing(flat_url)
        
        # Filter for actual image files
        image_files = [f for f in image_files if is_image_file(f)]
        
        if not image_files:
            print(f"  No images found in flat directory")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Download each image
        for i in range(50):
            image_file = image_files[i]
            image_url = urljoin(flat_url, image_file)
            save_path = os.path.join(DOWNLOAD_DIR, camera_name, image_file)
            
            if os.path.exists(save_path):
                total_skipped += 1
            else:
                time.sleep(DELAY_BETWEEN_REQUESTS)
                
            success = download_file(image_url, save_path)
            
            if success and not os.path.exists(save_path):
                total_failed += 1
            elif success:
                if not os.path.exists(save_path):
                    total_failed += 1
                else:
                    # Only count as downloaded if we actually downloaded it (not skipped)
                    if os.path.exists(save_path) and image_file not in [f for f in image_files if os.path.exists(os.path.join(DOWNLOAD_DIR, camera_name, 'images', 'flat', f))]:
                        total_downloaded += 1
            else:
                total_failed += 1
    
    # Final summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Total images downloaded: {total_downloaded}")
    print(f"Total images skipped (already exist): {total_skipped}")
    print(f"Total failed: {total_failed}")
    print(f"\nImages saved to: {os.path.abspath(DOWNLOAD_DIR)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
