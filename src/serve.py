"""
serve.py - FastAPI model serving application for CIFAR-10 classifier.

Endpoints:
  GET  /health   - Returns 200 if model is loaded and ready.
  POST /predict  - Accepts an image file, returns class probabilities.
"""

import io
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

# FastAPI & Uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_single_image_transform  # noqa: E402
from model import get_model, CIFAR10_CLASSES  # noqa: E402


# --------------------------------------------------------------------------- #
# App & global state
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="Serves a PyTorch ResNet-18 model trained on CIFAR-10.",
    version="1.0.0",
)

# Global model state
_model: Optional[torch.nn.Module] = None
_device: Optional[torch.device] = None
_transform = get_single_image_transform()


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model() -> None:
    """
    Load the trained model checkpoint from disk.
    Called once at startup.
    """
    global _model, _device

    checkpoint_path = os.environ.get(
        "MODEL_CHECKPOINT",
        "/app/checkpoints/classifier_v1.pt",
    )
    # Fallback for local development
    if not Path(checkpoint_path).exists():
        checkpoint_path = "checkpoints/classifier_v1.pt"

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(checkpoint_path).exists():
        print(
            f"[WARNING] Checkpoint not found at '{checkpoint_path}'. "
            "Model will not be loaded. /health will return 503.",
            flush=True,
        )
        return

    checkpoint = torch.load(checkpoint_path, map_location=_device)

    # Determine architecture from checkpoint config (if saved) or env var
    config = checkpoint.get("config", {})
    architecture = (
        config.get("model", {}).get("architecture")
        or os.environ.get("MODEL_ARCHITECTURE", "resnet18")
    )
    num_classes = (
        config.get("model", {}).get("num_classes")
        or int(os.environ.get("NUM_CLASSES", "10"))
    )

    _model = get_model(architecture=architecture, num_classes=num_classes, pretrained=False)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_device)
    _model.eval()

    print(
        f"[INFO] Model loaded: architecture={architecture}, "
        f"num_classes={num_classes}, device={_device}, "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Startup event
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def startup_event() -> None:
    load_model()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", summary="Health check")
async def health() -> JSONResponse:
    """
    Returns HTTP 200 with status 'ok' if the model is loaded and ready.
    Returns HTTP 503 if the model has not been loaded yet.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return JSONResponse(content={"status": "ok", "model_loaded": True})


@app.post("/predict", summary="Predict image class")
async def predict(image: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a PNG/JPEG image and return CIFAR-10 class probabilities.

    Request:
        multipart/form-data with field 'image' containing the image file.

    Response:
        JSON with 'predicted_class' and 'predictions' (sorted by probability).
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate content type
    if image.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {image.content_type}. Use PNG or JPEG.",
        )

    # Read and preprocess image
    try:
        raw_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = _transform(pil_image).unsqueeze(0).to(_device)  # (1, 3, 32, 32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}")

    # Inference
    with torch.no_grad():
        logits = _model(tensor)                          # (1, 10)
        probabilities = F.softmax(logits, dim=1)[0]     # (10,)

    # Build response
    probs_list = probabilities.cpu().tolist()
    predictions = [
        {"class": cls, "probability": round(prob, 4)}
        for cls, prob in zip(CIFAR10_CLASSES, probs_list)
    ]
    predictions_sorted = sorted(predictions, key=lambda x: x["probability"], reverse=True)
    predicted_class = predictions_sorted[0]["class"]

    return JSONResponse(content={
        "predicted_class": predicted_class,
        "predictions": predictions_sorted,
    })


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("serve:app", host="0.0.0.0", port=port, reload=False)
