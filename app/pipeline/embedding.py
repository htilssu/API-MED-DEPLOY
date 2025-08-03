import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import google.generativeai as genai
from huggingface_hub import login
import timm
from PIL import Image
import logging

# ---------------------- CẤU HÌNH ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------- CONSTANTS ----------------------
PROCESSED_DIR = "app/static/processed"
ANOMALY_MAP_DIR = "app/static/anomaly_maps"
ROI_OUTPUT_DIR = "app/static/roi_outputs"
GCS_BUCKET = "group_dataset-nt"
GCS_FOLDER = "handle_data"
LOCAL_SAVE_DIR = "app/static/"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
vit_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image_clip(image_pil: Image.Image) -> np.ndarray:
   
    try:
        # Ensure image is in RGB format
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        # Process image for CLIP model
        inputs = processor(images=image_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        
        # Convert outputs to NumPy array
        image_embedding = outputs.cpu().numpy().astype("float32")
        return image_embedding
    
    except Exception as e:
        raise ValueError(f"Không thể xử lý ảnh: {str(e)}")