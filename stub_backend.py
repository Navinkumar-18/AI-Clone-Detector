# stub_backend.py — temporary, swap out once real model is ready
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev/demo, don't ship this to prod
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # TEMP: fake response — replace with real predict() call later
    is_spoof = random.random() > 0.5
    confidence = random.uniform(0.7, 0.99)
    return {
        "label": "spoof" if is_spoof else "bonafide",
        "confidence": round(confidence, 4),
        "risk_level": "high" if is_spoof and confidence > 0.85 else "medium" if is_spoof else "low"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}