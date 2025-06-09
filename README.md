# Flickd Smart Fashion Tagger

This project is a backend pipeline for smart fashion product tagging and vibe classification from short videos (like Instagram Reels). It uses trained object detection (YOLOv8m), fashion-specific image embeddings (FashionCLIP), FAISS for fast similarity search, and NLP for vibe prediction.

---

## 🚀 Setup Instructions

### 1. Clone and Install Dependencies

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
### 2. Clone and Install FashionCLIP

```bash
git clone https://github.com/patrickjohncyh/fashion-clip.git
cd fashion-clip
pip install -e .
```
⚠️ This installs the FashionCLIP model.

## 🎬 How to Run
### 1. Set your video path in `main.py` :
```python
video_path = "videos/reel_001.mp4"
```

### 2. Run the script:
```bash
python main.py
```
Ensure you also have a corresponding caption file (e.g., `videos/reel_001.txt`) and a product data CSV (`data/product_data.csv`).

# Pipeline Breakdown

### Step 1: Frame Extraction
- The input video is split into frames at a configurable rate (`frame_rate=5`).
- Frames are saved in the `frames/` directory.

### Step 2: Object Detection (YOLOv8)
- Each frame is passed through a custom-trained YOLOv8 model.
- Detected fashion items (e.g. dresses, tops) are cropped and saved to `detections/`.
### Step 3: Product Matching (FashionCLIP + FAISS)
- Each detected crop is embedded using FashionCLIP.
- It's compared against a product catalog index using cosine similarity.
- Products above a confidence threshold (≥ 0.75) are considered valid matches.
### Step 4: Metadata Mapping
- Matching product info is extracted from a CSV (`product_data.csv`) including:
    - Product ID
    - Color (from tags)
    - Product Type
### Step 5: Vibe Classification (NLP)
- The script reads the corresponding caption text (e.g., `reel_001.txt`)
- It classifies the vibe using `facebook/bart-large-mnli` zero-shot classification from HuggingFace.
### Step 6: Output Generation
- The final structured output is saved in `outputs/{video_id}.json`
```json
{
    "video_id": "reel_002",
    "vibes": [
        "Coquette",
        "attention-grabbing",
        "Vintage"
    ],
    "products": [
        {
            "type": "Dress",
            "color": "White",
            "matched_product_id": "118458",
            "match_type": "similar",
            "confidence": 0.756
        }
    ]
}
```
