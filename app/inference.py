# Library setup
import torch
import numpy as np
from PIL import Image
from model.unet import UNet

# Load the model first
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model/unet_epoch20.pth"

model = UNet(in_channels=1, out_channels=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

## step 1: Preprocessing
def preprocess_image(img: Image.Image):
    gray = img.convert("L").resize((128, 128))
    array = np.array(gray).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, gray

## step 2: Segmentation
def segment(img: Image.Image):
    tensor, base_img = preprocess_image(img)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
    # Overlay colors
    overlay = np.zeros((128, 128, 3), dtype=np.uint8)
    colors = {1: [255, 0, 0], 2: [0, 255, 0]} # red/green
    for cls, color in colors.items():
        overlay[pred == cls] = color

    blended = np.stack([np.array(base_img)]*3, axis=-1)
    blended = (0.5 * blended + 0.5 * overlay).astype(np.uint8)
    return Image.fromarray(blended), pred