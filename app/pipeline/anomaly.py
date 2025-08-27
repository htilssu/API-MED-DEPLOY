import cv2
import numpy as np
import torch
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import login
import timm
from PIL import Image
import logging
from app.pipeline.embedding import embed_image_clip
from app.config.setting import setting

# ---------------------- CẤU HÌNH ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
login(token=setting.HUGGINGFACE_TOKEN)

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


def generate_anomaly_overlay(image_pil):
    image_resized = image_pil.resize((224, 224))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_np = (image_np - 0.5) / 0.5
    image_tensor = torch.tensor(image_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        features = vit_model.forward_features(image_tensor)

    anomaly_map = features[0].norm(dim=0).cpu().numpy()
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    anomaly_map_resized = cv2.resize(anomaly_map, image_pil.size[::-1])
    anomaly_map_blur = cv2.GaussianBlur(anomaly_map_resized, (5, 5), 0)
    heatmap = cv2.applyColorMap(anomaly_map_blur, cv2.COLORMAP_JET)

    image_cv = np.array(image_pil)
    if heatmap.shape[:2] != image_cv.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_cv.shape[1], image_cv.shape[0]))

    if image_cv.shape[2] == 3:
        original = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    else:
        original = image_cv

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return overlay, anomaly_map_blur

def save_anomaly_outputs(anomaly_overlay, anomaly_map, image_path: str):
    basename = Path(image_path).stem
    roi_output_path = os.path.join(ROI_OUTPUT_DIR, f"{basename}_overlay.jpg")
    anomaly_map_path = os.path.join(ANOMALY_MAP_DIR, f"{basename}_anomaly.jpg")

    os.makedirs(ROI_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANOMALY_MAP_DIR, exist_ok=True)

    cv2.imwrite(roi_output_path, anomaly_overlay)
    cv2.imwrite(anomaly_map_path, anomaly_map)

    return roi_output_path, anomaly_map_path

def embed_anomaly_heatmap(heatmap_path: str) -> np.ndarray:
    image = Image.open(heatmap_path).convert("RGB")
    return embed_image_clip(image)