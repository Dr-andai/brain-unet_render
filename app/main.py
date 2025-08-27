from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from PIL import Image
import time

from app.inference import segment

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EXAMPLE_DIR = STATIC_DIR / "examples"
RESULT_PATH = STATIC_DIR / "result.png"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    example_images = [f.name for f in EXAMPLE_DIR.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    return templates.TemplateResponse("index.html", {"request": request, "example_images": example_images})

@app.post("/segment", response_class=HTMLResponse)
async def run_segmentation(
    request: Request,
    model: str = Form(...),
    image_name: str = Form(...)
):
    # Resolve paths
    model_path = Path("model") / model
    image_path = EXAMPLE_DIR / image_name

    # Load image
    img = Image.open(image_path)

    # Run segmentation
    output_img, pred = segment(img, model_path)

    # Save result
    output_img.save(RESULT_PATH)

    # Stats
    total = pred.size
    stats = {
        "Background": round((pred == 0).sum() / total, 2),
        "Gray Matter": round((pred == 1).sum() / total, 2),
        "White Matter": round((pred == 2).sum() / total, 2),
    }

    result_url = f"/static/result.png?ts={int(time.time())}"

    return templates.TemplateResponse(
        "partials/result.html",
        {"request": request, "result": result_url, "stats": stats}
    )
