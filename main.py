import cv2
import os
from ultralytics import YOLO
import json
import numpy as np
import faiss
from PIL import Image
import requests
from io import BytesIO
from fashion_clip.fashion_clip import FashionCLIP
import torch
import random
import shutil
import pandas as pd
from transformers import pipeline

# Load FAISS index and product IDs
index = faiss.read_index("data/fashionclip_catalog.index")
product_ids = np.load("data/fashionclip_ids.npy", allow_pickle=True)

# Load FashionCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
fclip = FashionCLIP('fashion-clip')
fclip.model.to(device)

# Load pretrained YOLOv8 model
model_path = 'models\\runs\\detect\\train\\weights\\best.pt'
model = YOLO(model_path)

def extract_frames(video_path, output_dir="frames/", frame_rate=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_id % frame_rate == 0:
            filename = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        frame_id += 1
    cap.release()

def detect_on_frames(frame_dir, crop_dir="detections/"):
    if os.path.exists(crop_dir):
        shutil.rmtree(crop_dir)
    os.makedirs(crop_dir, exist_ok=True)
    count = 0
    # detections = []

    for filename in sorted(os.listdir(frame_dir)):
        if not filename.endswith(".jpg"):
            continue

        frame_path = os.path.join(frame_dir, filename)
        frame_num = int(filename.split("_")[-1].split(".")[0])

        results = model.predict(frame_path, conf=0.4, imgsz=640)[0]

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            # Get binding box in xywh format
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Crop and save image
            image = cv2.imread(frame_path)
            crop = image[y1:y2, x1:x2]
            crop_filename = os.path.join(crop_dir, f"{frame_num}_{i}_{class_name}.jpg")
            cv2.imwrite(crop_filename, crop)

            count+=1

            # detections.append({
            #     "frame": frame_num,
            #     "class": class_name,
            #     "bbox": [x1, y1, w, h],
            #     "confidence": round(conf, 3),
            #     "crop_path": crop_filename
            # })

    # # Save detections as JSON
    # with open(output_json, "w") as f:
    #     json.dump(detections, f, indent=2)

    print(f"✅ Processed {count} detections.")

def get_match_type(score):
    if score >= 0.9:
        return "exact"
    elif score >= 0.735:
        return "similar"
    else:
        return "none"

def search_by_local_image(image_path, top_k=1):
    try:
        image = Image.open(image_path).convert("RGB")
        img_emb = fclip.encode_images([image], batch_size=1)[0].astype("float32")
        faiss.normalize_L2(img_emb.reshape(1, -1))
        D, I = index.search(img_emb.reshape(1, -1), top_k)
        results = [(product_ids[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
        return results
    except Exception as e:
        print(f"Could not process image: {image_path}, error: {e}")
        return []

def classify_vibes(text, num_vibes=3):
    result = classifier(text, extended_labels, multi_label=True)
    scores = list(zip(result["labels"], result["scores"]))

    # Top sorted vibes
    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)

    # Collect vibes
    top_labels = [label for label, _ in top_matches[:num_vibes]]

    # Separate primary and fallback
    primary = [v for v in top_labels if v in vibe_labels]
    fallback = [v for v in top_labels if v not in vibe_labels]

    # Ensure at least one primary vibe
    if not primary:
        primary = [label for label, _ in top_matches if label in vibe_labels][:1]

    # Fill up to 3 with other unique labels
    final = list(dict.fromkeys(primary + fallback))  # preserve order + remove dupes
    while len(final) < num_vibes:
        pool = [v for v in extended_labels if v not in final]
        final.append(random.choice(pool))

    return final[:num_vibes]

video_path = "videos/reel_003.mp4" #set your video path here <------
extract_frames(video_path)

base_name = os.path.basename(video_path)
file_name_without_extension, file_extension = os.path.splitext(base_name)
video_id = file_name_without_extension

detect_on_frames("frames/")

results_json = []
detection_dir = "detections"
best_match_map = {}  # key = matched_product_id, value = (score, full_result_dict)

df = pd.read_csv("data/product_data.csv")


for filename in os.listdir(detection_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(detection_dir, filename)
        matches = search_by_local_image(image_path, top_k=1)

        for pid, score in matches:
            if score >= 0.735:  # Only consider matches with score above 0.735
                pid_str = str(pid)
                match_type = get_match_type(score)
                
                # If this product hasn't been matched yet or new score is better
                if pid_str not in best_match_map or score > best_match_map[pid_str]["confidence"]:
                    color_data = df[df["id"] == pid]["product_tags"].values[0]
                    fields = [field.strip() for field in color_data.split(',')]
                    color = None
                    for field in fields:
                        if field.lower().startswith("colour:"):
                            color = field.split(":", 1)[1]
                            break


                    f_type = df[df["id"] == pid]["product_type"].values[0]
                    best_match_map[pid_str] = {
                        "type": str(f_type),
                        "color": str(color),
                        "matched_product_id": pid_str,
                        "match_type": match_type,
                        "confidence": round(score, 3)
                    }

# Convert map to final list
results_json = list(best_match_map.values())

#vibe classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
vibe_labels = [
    "Coquette", "Clean Girl", "Cottagecore", "Streetcore", "Y2K", "Boho", "Party Glam"
]
# Extended label list
extended_labels = vibe_labels + [
    "Evening", "Romantic", "Retro", "Urban", "Minimalist", "Hyper-feminine",
     "Chic", "Edgy", "Vintage", "Modern", "Formal", "Casual",
     "attention-grabbing", "Bold", "nostalgic", "Indie Sleaze", "Glamorous",
     "Gorpcore", "Mermaidcore", "Dark Academia", "Light Academia", "Fairycore",
]

with open(f"videos/{video_id}.txt", "r", encoding="utf-8") as file:
    caption = file.read().strip()

vibes = classify_vibes(caption)

final_output = {
                    "video_id": video_id,
                    "vibes": vibes,
                    "products": results_json
                }

with open(f"outputs/{video_id}.json", "w") as f: 
    json.dump(final_output, f, indent=4)

print(f"Saved {len(results_json)} matches to 'outputs/{video_id}.json'")