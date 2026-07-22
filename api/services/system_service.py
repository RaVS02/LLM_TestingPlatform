"""
Wykrywanie dostepnych zasobow sprzetowych (RAM + VRAM) i ocena, czy dany
model bezpiecznie zmiesci sie w pamieci, czy jest ryzykowny/za duzy.

Uwaga: VRAM wykrywamy na razie tylko dla NVIDIA (przez nvidia-smi), bo to
najbardziej ujednolicone API. AMD (rocm-smi) / Intel mozna dopisac pozniej,
jesli beda potrzebne - na razie brak nvidia-smi = zakladamy inference na CPU/RAM.
"""

import re
import shutil
import subprocess

import psutil

def get_ram_info() -> dict:
    vm = psutil.virtual_memory()
    return {"total": vm.total, "available": vm.available}


def get_vram_info() -> list:
    """Zwraca liste GPU: [{"total": bajty, "free": bajty}, ...].
    Pusta lista, jesli nie znaleziono NVIDIA GPU / brak nvidia-smi w PATH."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        wynik = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if wynik.returncode != 0 or not wynik.stdout.strip():
        return []

    gpus = []
    for linia in wynik.stdout.strip().splitlines():
        try:
            total_mb, free_mb = [int(x.strip()) for x in linia.split(",")]
        except ValueError:
            continue
        gpus.append({"total": total_mb * 1024 * 1024, "free": free_mb * 1024 * 1024})
    return gpus


def get_system_resources() -> dict:
    ram = get_ram_info()
    gpus = get_vram_info()
    return {
        "ram_total": ram["total"],
        "ram_available": ram["available"],
        "vram_total": sum(g["total"] for g in gpus),
        "vram_available": sum(g["free"] for g in gpus),
        "gpu_count": len(gpus),
    }


_JEDNOSTKI = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}


def parsuj_rozmiar_do_bajtow(tekst) -> int:
    """Parsuje string typu '5.5GB', '200 MB' na bajty. Zwraca None jesli sie nie da."""
    if not tekst:
        return None
    dopasowanie = re.match(r"^\s*([\d.,]+)\s*([KMGT]?B)\s*$", str(tekst).strip(), re.IGNORECASE)
    if not dopasowanie:
        return None
    try:
        liczba = float(dopasowanie.group(1).replace(",", "."))
    except ValueError:
        return None
    jednostka = dopasowanie.group(2).upper()
    return int(liczba * _JEDNOSTKI.get(jednostka, 1))


# Margines bezpieczenstwa - Ollama w praktyce potrzebuje troche wiecej pamieci
# niz "goly" rozmiar pliku modelu (KV-cache kontekstu, bufor na dekodowanie).
MARGINES_BEZPIECZENSTWA = 1.20


def ocen_ryzyko(model_size_bytes: int, zasoby: dict) -> dict:
    """Zwraca {"status": "bezpieczny"|"ryzykowny"|"za_duzy"|"nieznany", "powod": str}."""
    if not model_size_bytes:
        return {"status": "nieznany", "powod": "nie udalo sie ustalic rozmiaru modelu"}

    wymagane = model_size_bytes * MARGINES_BEZPIECZENSTWA
    ram_dostepny = zasoby.get("ram_available", 0)
    vram_dostepny = zasoby.get("vram_available", 0)
    ma_gpu = zasoby.get("gpu_count", 0) > 0

    if ma_gpu and wymagane <= vram_dostepny:
        return {"status": "bezpieczny", "powod": "miesci sie w wolnym VRAM z zapasem"}

    if wymagane <= ram_dostepny + vram_dostepny:
        powod = (
            "miesci sie w RAM+VRAM z zapasem, ale czesc/calosc pojdzie na CPU (wolniej)"
            if ma_gpu else
            "miesci sie w dostepnym RAM z zapasem (inference na CPU - wolniej)"
        )
        return {"status": "ryzykowny", "powod": powod}

    return {
        "status": "za_duzy",
        "powod": f"model potrzebuje ok. {wymagane / 1024**3:.1f} GB z zapasem, "
                 f"dostepne to {(ram_dostepny + vram_dostepny) / 1024**3:.1f} GB (RAM+VRAM)",
    }
