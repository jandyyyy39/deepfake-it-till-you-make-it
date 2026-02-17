# Deepfake It Till You Make It: Project Phantasm

**Goal:** Train an ensemble of forensic experts to detect GenAI images, then stress-test them by injecting real camera physics (PRNU) into the AI data.

---

## Prerequisites

We use a Conda environment to manage dependencies.

**1. Create the Environment**
Run the following command to install all necessary libraries (PyTorch, OpenCV, NumPy, etc.) from the `environment.yml` file:
```bash
conda env create -f environment.yml

conda activate deepfake_env
```
## Phase 1: Dataset Acquisition

We utilize two primary datasets for this project.
Sources:  
* GenImage (AI & Nature): [Google Drive](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS)  
* VISION (Real Camera Forensics): [VISION](https://lesc.dinfo.unifi.it/VISION/dataset/)  
  
### Downloading the GenImage Splits
We have a helper script to automate downloading the specific generator splits (located in the `scripts/` directory).

Arguments:
'adm', 'biggan', 'glide', 'midjourney', 'sdv4', 'sdv5', 'vqdm', 'wukong'  
  
### Usage
```python download_genimage_dataset.py --model sdv5```  
  
### Important: "Quota Exceeded" Error
If the script fails with a Google Drive "Quota Exceeded" error, you must download the split manually.  
1. Go to the GenImage Google Drive.  
2. Right-click the file -> "Make a Copy" to your own Drive.  
3. Download your copy.  
  
## Phase 2: Signal Extraction (Forensics)

To simulate real camera sensors, we extract the Photo Response Non-Uniformity (PRNU) noise from the VISION dataset. This "fingerprint" will later be injected into AI images to test our detectors.
Run the Extraction
```python scripts/extract_pnru_fingerprints.py```  

This code will download the first 10 flat images of each camera type in the VISION dataset.
