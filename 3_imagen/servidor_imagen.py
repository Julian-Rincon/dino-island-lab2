"""
Servidor text-to-image para el dinosaurio (aMUSEd 512).

Recibe   POST /generar  {"prompt": str, "seed": int (opcional), "pasos": int (opcional)}
Devuelve {"png_base64": str, "segundos": float, "modelo": "amused/amused-512", "dispositivo": "cuda"|"cpu"}
GET /salud -> {"ok": true, ...}

Corre igual en Colab (GPU) que en el notebook de SageMaker (CPU):
    uvicorn servidor_imagen:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import os
import time

import torch
from diffusers import AmusedPipeline
from fastapi import FastAPI
from pydantic import BaseModel

MODELO = os.environ.get("MODELO_IMAGEN", "amused/amused-512")
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DISPOSITIVO == "cuda" else torch.float32

if DISPOSITIVO == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

pipe = AmusedPipeline.from_pretrained(
    MODELO, variant="fp16" if DTYPE == torch.float16 else None, torch_dtype=DTYPE
).to(DISPOSITIVO)
if DTYPE == torch.float16:
    pipe.vqvae.to(torch.float32)   # el VQ-GAN en fp16 da artefactos

NEGATIVO = "blurry, low quality, text, watermark, deformed, extra limbs, cartoon, cropped"

app = FastAPI(title="Generador de imagen del dinosaurio")


class Pedido(BaseModel):
    prompt: str
    seed: int = 7
    pasos: int = 12


def generar_png(prompt: str, seed: int = 7, pasos: int = 12) -> bytes:
    g = torch.Generator(DISPOSITIVO).manual_seed(seed)
    img = pipe(prompt, negative_prompt=NEGATIVO, num_inference_steps=pasos,
               guidance_scale=10.0, generator=g).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/salud")
def salud():
    return {"ok": True, "modelo": MODELO, "dispositivo": DISPOSITIVO}


@app.post("/generar")
def generar(p: Pedido):
    t0 = time.time()
    png = generar_png(p.prompt[:500], p.seed, max(4, min(p.pasos, 24)))
    return {"png_base64": base64.b64encode(png).decode(), "segundos": round(time.time() - t0, 1),
            "modelo": MODELO, "dispositivo": DISPOSITIVO}
