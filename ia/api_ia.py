
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

app = FastAPI(title="AccessOne - IA")

# Caminho do modelo que você já treinou
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "train-3"
    / "weights"
    / "best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modelo não encontrado: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


@app.get("/")
def inicio():
    return {"mensagem": "API da IA AccessOne funcionando!"}


@app.post("/predict")
async def reconhecer_produto(
    file: UploadFile = File(...)
):
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp"
    ):
        raise HTTPException(
            status_code=400,
            detail="Envie uma imagem JPG, PNG ou WEBP."
        )

    try:
        conteudo = await file.read()
        imagem = Image.open(
            BytesIO(conteudo)
        ).convert("RGB")

        imagem_np = np.array(imagem)

        resultados = model.predict(
            source=imagem_np,
            conf=0.50,
            verbose=False
        )

        deteccoes = []
        resultado = resultados[0]

        for caixa in resultado.boxes:
            classe_id = int(caixa.cls.item())
            confianca = float(caixa.conf.item())

            deteccoes.append({
                "codigo": model.names[classe_id],
                "confianca": round(confianca, 4)
            })

        if not deteccoes:
            return {
                "encontrado": False,
                "mensagem": "Não consegui identificar o produto."
            }

        melhor = max(
            deteccoes,
            key=lambda item: item["confianca"]
        )

        return {
            "encontrado": True,
            "produto": melhor["codigo"],
            "confianca": melhor["confianca"],
            "deteccoes": deteccoes
        }

    except HTTPException:
        raise
    except Exception as erro:
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao analisar a imagem: {erro}"
        )
