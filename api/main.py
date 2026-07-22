import os
import shutil
import uuid
from typing import List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from funkcje import load_models, create_raport
from api.services import ollama_service, hf_service, system_service

app = FastAPI(title="LLM Vision Tests API")

UPLOAD_DIR = "uploads"
RAPORT_PATH = os.path.join("raports", "raport.csv")
CSV_BACKEND = {
    "ollama": "llm_models/ollama_models.csv",
    "hf": "llm_models/hf_models.csv",
}

os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/models")
def get_models(backend: str):
    if backend not in CSV_BACKEND:
        raise HTTPException(400, "backend musi byc 'ollama' albo 'hf'")
    try:
        df = load_models(CSV_BACKEND[backend])
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    rekordy = df.to_dict(orient="records")
    zasoby = system_service.get_system_resources()

    if backend == "ollama":
        for rekord in rekordy:
            rekord["capabilities"] = ollama_service.get_capabilities(rekord["model_name"])
            rozmiar_bajty = ollama_service.get_model_size_bytes(rekord["model_name"])
            if rozmiar_bajty is None:
                # Model jeszcze nie pobrany (albo Ollama chwilowo niedostepna) -
                # przyblizamy na podstawie kolumny "size" w CSV.
                rozmiar_bajty = system_service.parsuj_rozmiar_do_bajtow(rekord.get("size"))
            rekord["hardware"] = system_service.ocen_ryzyko(rozmiar_bajty, zasoby)
    else:
        # Pipeline'y HF w tym projekcie sa na sztywno "image-text-to-text" (hf_service.py),
        # wiec zakladamy vision - transformers nie ma odpowiednika /api/show do sprawdzenia tego.
        for rekord in rekordy:
            rekord["capabilities"] = ["vision"]
            rozmiar_bajty = system_service.parsuj_rozmiar_do_bajtow(rekord.get("size"))
            rekord["hardware"] = system_service.ocen_ryzyko(rozmiar_bajty, zasoby)

    return rekordy


@app.get("/system/resources")
def system_resources():
    """RAM/VRAM aktualnie dostepne na maszynie hostujacej API - do wyswietlenia
    w UI oraz jako baza do oceny ryzyka per model w /models."""
    return system_service.get_system_resources()


@app.post("/models/refresh-capabilities")
def refresh_capabilities():
    """Czysci cache capabilities i rozmiarow Ollamy - przydatne po `ollama pull`
    nowego modelu albo po zmianie CSV bez restartu API."""
    ollama_service.clear_capabilities_cache()
    ollama_service.clear_size_cache()
    return {"status": "ok"}


import math

def _nan_to_none(value):
    """pandas zwraca puste liczbowe komorki jako float('nan'), ktory nie jest
    poprawnym JSON-em i wywala renderowanie odpowiedzi FastAPI. Zamieniamy na None (-> null)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@app.get("/raport")
def get_raport():
    if not os.path.isfile(RAPORT_PATH):
        return []
    try:
        df = pd.read_csv(RAPORT_PATH)
        rekordy = df.to_dict(orient="records")
        rekordy = [{k: _nan_to_none(v) for k, v in row.items()} for row in rekordy]
    except Exception as e:
        raise HTTPException(500, f"Nie udalo sie odczytac raports/raport.csv: {e}")
    return rekordy


def _resolve_model_ref(backend: str, model_id: int) -> str:
    df = load_models(CSV_BACKEND[backend])
    row = df[df["model_id"] == model_id]
    if row.empty:
        raise HTTPException(404, f"Nie znaleziono modelu o id {model_id} dla backendu {backend}")
    if backend == "ollama":
        return row["model_name"].values[0]
    return row["path"].values[0]


def _save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1]
    dest = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


@app.post("/generate")
async def generate(
    backend: str = Form(...),
    model_id: int = Form(...),
    system_prompt: str = Form(""),
    user_prompt: str = Form("Describe this image in a highly technical and analytical way, keeping it concise"),
    temperature: float = Form(0.8),
    top_p: float = Form(0.9),
    max_tokens: int = Form(200),
    ctx_len: int = Form(4096),
    images: List[UploadFile] = File(...),
):
    if backend not in CSV_BACKEND:
        raise HTTPException(400, "backend musi byc 'ollama' albo 'hf'")

    model_ref = _resolve_model_ref(backend, model_id)
    service = ollama_service if backend == "ollama" else hf_service

    dodatkowe_argumenty = {"ctx_len": ctx_len} if backend == "ollama" else {}

    # Przetwarzanie sekwencyjne, jeden obraz po drugim - bezpieczniejsze dla RAM/GPU.
    wyniki = []
    for image in images:
        image_path = _save_upload(image)
        try:
            opis = service.generate(
                model_ref,
                image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **dodatkowe_argumenty,
            )
            create_raport(
                image.filename,
                model_ref,
                opis,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            wyniki.append({"plik": image.filename, "opis": opis, "blad": None})
        except Exception as e:
            wyniki.append({"plik": image.filename, "opis": None, "blad": str(e)})

    return JSONResponse({"backend": backend, "model": model_ref, "wyniki": wyniki})
