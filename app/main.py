import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import os

from app.inference import segment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = "static"

os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/segment", response_class=HTMLResponse)
async def run_segmentation(request: Request, file: UploadFile = File(...)):
    if not (file.filename.lower().endswith(".png") or file.filename.lower().endswith(".jpg") or file.filename.lower().endswith(".jpeg")):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Only .png or .jpg images are supported."},
            )
    
    img = Image.open(io.BytesIO(await file.read()))
    output_img, pred = segment(img)

    # Stats
    total = pred.size
    stats = {
        "Background": round((pred == 0).sum() / total, 2),
        "Gray Matter": round((pred == 1).sum() / total, 2),
        "White Matter": round((pred == 2).sum() / total, 2),
        }
    
    # Save result temporarily
    result_filename = "result.png"
    result_path = os.path.join(STATIC_DIR, result_filename)
    output_img.save(result_path)
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request,
         "result": f"/static/{result_filename}",
        "stats": stats,},
        )