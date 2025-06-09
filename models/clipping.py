# build_catalog_embeddings.py
import pandas as pd
import clip
import torch
import requests
from PIL import Image
import io
import faiss
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

df = pd.read_csv("../data/images.csv")
catalog_embeddings = []
product_ids = []

for i, row in df.iterrows():
    try:
        response = requests.get(row["image_url"], timeout=5)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input).cpu().numpy()
        catalog_embeddings.append(embedding[0])
        product_ids.append(row["id"])
    except Exception as e:
        print(f"Failed to process {row['image_url']}: {e}")

# Save FAISS index
dimension = catalog_embeddings[0].shape[0]
index = faiss.IndexFlatIP(dimension)
catalog_matrix = np.array(catalog_embeddings).astype("float32")
faiss.normalize_L2(catalog_matrix)  # normalize for cosine similarity
index.add(catalog_matrix)

faiss.write_index(index, "data/catalog.index")
np.save("data/product_ids.npy", np.array(product_ids))
