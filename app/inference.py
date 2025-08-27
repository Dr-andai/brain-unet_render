import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Libraries
import torch
import numpy as np
from PIL import Image
from model.unet import UNet

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache to avoid reloading models repeatedly
_model_cache = {}

def load_model(model_path: Path):
    """
    Load a U-Net model from a .pth file, cached for reuse.
    """
    model_path = str(model_path)

    if model_path in _model_cache:
        return _model_cache[model_path]

    model = UNet(in_channels=1, out_channels=3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _model_cache[model_path] = model
    return model


## Step 1: Preprocessing
def preprocess_image(img: Image.Image):
    gray = img.convert("L").resize((128, 128))
    array = np.array(gray).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, gray


## Step 2: Segmentation
def segment(img: Image.Image, model_path: Path):
    """
    Run segmentation with the model specified by `model_path`.
    """
    model = load_model(model_path)
    tensor, base_img = preprocess_image(img)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Create overlay
    overlay = np.zeros((128, 128, 3), dtype=np.uint8)
    colors = {
        1: [255, 0, 0],  # red → Gray Matter
        2: [0, 255, 0],  # green → White Matter
    }
    for cls, color in colors.items():
        overlay[pred == cls] = color

    # Blend overlay with base image
    blended = np.stack([np.array(base_img)] * 3, axis=-1)
    blended = (0.5 * blended + 0.5 * overlay).astype(np.uint8)

    return Image.fromarray(blended), pred
