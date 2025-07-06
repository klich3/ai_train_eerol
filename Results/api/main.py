#!/usr/bin/env python3
# 🦷 API de Inferencia Dental AI

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import uvicorn

app = FastAPI(
    title="🦷 Dental AI API",
    description="API para análisis dental con deep learning",
    version="1.0.0"
)

# Configuración global
MODELS_DIR = Path("../models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache de modelos cargados
loaded_models = {}

@app.get("/")
async def root():
    return {
        "message": "🦷 Dental AI API",
        "version": "1.0.0",
        "status": "active",
        "device": str(DEVICE)
    }

@app.get("/models")
async def list_models():
    """Lista modelos disponibles."""
    models = {
        "detection": list((MODELS_DIR / "yolo_detect").glob("*.pt")) if (MODELS_DIR / "yolo_detect").exists() else [],
        "segmentation": list((MODELS_DIR / "yolo_segment").glob("*.pt")) if (MODELS_DIR / "yolo_segment").exists() else [],
        "classification": list((MODELS_DIR / "cnn_classifier").glob("*.pth")) if (MODELS_DIR / "cnn_classifier").exists() else []
    }
    
    return {
        "available_models": {k: [str(m.name) for m in v] for k, v in models.items()},
        "total_models": sum(len(v) for v in models.values())
    }

@app.post("/predict/detection")
async def predict_detection(
    file: UploadFile = File(...),
    model_name: str = "dental_dataset_best.pt",
    confidence: float = 0.25
):
    """Detección de estructuras dentales."""
    try:
        # Cargar imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Aquí iría la lógica de inferencia YOLO
        # Por ahora retornamos un ejemplo
        
        return {
            "predictions": [
                {
                    "class": "tooth",
                    "confidence": 0.89,
                    "bbox": [100, 150, 200, 250]
                },
                {
                    "class": "caries",
                    "confidence": 0.67,
                    "bbox": [180, 200, 220, 240]
                }
            ],
            "model_used": model_name,
            "image_size": image.size,
            "processing_time": "0.45s"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/classification")
async def predict_classification(
    file: UploadFile = File(...),
    model_name: str = "dental_dataset_best.pth"
):
    """Clasificación de patologías dentales."""
    try:
        # Cargar imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Aquí iría la lógica de inferencia CNN
        # Por ahora retornamos un ejemplo
        
        return {
            "prediction": {
                "class": "caries",
                "confidence": 0.87,
                "probabilities": {
                    "normal": 0.13,
                    "caries": 0.87,
                    "filling": 0.00
                }
            },
            "model_used": model_name,
            "image_size": image.size,
            "processing_time": "0.23s"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Verificación de salud del API."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": len(loaded_models)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
