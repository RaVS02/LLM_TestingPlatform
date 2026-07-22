# LLM Vision Tests

Aplikacja do lokalnego testowania i porównywania modeli multimodalnych (Vision-to-Text) — Ollama i HuggingFace. REST API (FastAPI) + prosty interfejs w przeglądarce: wybór backendu/modelu, prompt systemowy + własne uwagi, parametry generowania, pojedyncze zdjęcie lub cały folder naraz. Wyniki zapisują się do `raports/raport.csv`.

Działa **natywnie** (bez Dockera) — pomyślana pod pracę na kilku komputerach (np. laptop + stacjonarny), z ręcznym przenoszeniem pobranych modeli między nimi przez dysk zewnętrzny.

---

## Struktura projektu

```
llm_vision_tests/
├── requirements.txt
├── funkcje.py                  ← wspolne funkcje (CSV modeli, zapis raportu)
├── api/
│   ├── main.py                 ← REST API (FastAPI)
│   └── services/
│       ├── ollama_service.py
│       └── hf_service.py
├── static/
│   └── index.html              ← interfejs w przegladarce
├── llm_models/
│   ├── ollama_models.csv       ← lista modeli Ollama (edytuj recznie)
│   └── hf_models.csv           ← lista modeli HuggingFace (edytuj recznie)
├── raports/raport.csv          ← historia wygenerowanych opisow
├── images/                     ← Twoje zdjecia testowe
└── uploads/                    ← tymczasowe pliki wgrane przez interfejs
```

**Modele Ollama NIE są częścią tego folderu** — mieszkają osobno w `~/.ollama/models` (Linux/macOS) albo `%USERPROFILE%\.ollama\models` (Windows). Kod projektu = lekki, kilka MB. Modele = ciężkie, przenosisz je osobno (patrz niżej).

---

## Instalacja na każdym komputerze (identyczne kroki laptop/desktop)

1. **Zainstaluj Ollamę:** [ollama.com/download](https://ollama.com/download) — startuje jako usługa w tle automatycznie po instalacji.

2. **Środowisko Python:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   *(uwaga: jeśli sam `pip ...` nie działa mimo aktywnego venv, zawsze używaj `python -m pip ...`)*

3. **Pobierz modele** (albo skopiuj z drugiego komputera — patrz sekcja niżej):
   ```powershell
   ollama pull llama3.2-vision
   ollama pull moondream
   ```

4. **Uruchom:**
   ```powershell
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Otwórz: `http://localhost:8000` (dokumentacja API: `http://localhost:8000/docs`)

Aplikacja łączy się z Ollamą pod `http://localhost:11434` (zmienna `OLLAMA_HOST`, jeśli chcesz to nadpisać). GPU (NVIDIA/CUDA) dla modeli HuggingFace wykrywane jest automatycznie — bez GPU liczy na CPU.

---

## 🔄 Przenoszenie pobranych modeli między komputerami

Zamiast ściągać te same, duże modele na każdym komputerze osobno, możesz skopiować gotowe pliki:

**Gdzie Ollama trzyma modele:**
- Windows: `%USERPROFILE%\.ollama\models`
- Linux/macOS: `~/.ollama/models`

**Struktura w środku:**
```
models/
├── manifests/    ← male pliki opisujace wersje/tagi modeli
└── blobs/        ← duze pliki binarne z wagami (rzeczywisty rozmiar modelu)
```

**Kopiowanie (np. na dysk zewnętrzny i z powrotem):**
```powershell
# Z komputera zrodlowego na dysk zewnetrzny
robocopy "$env:USERPROFILE\.ollama\models" "D:\ollama-models-backup" /E

# Na docelowym komputerze - z dysku do lokalnego folderu Ollamy
robocopy "D:\ollama-models-backup" "$env:USERPROFILE\.ollama\models" /E
```
(na Linux/macOS: zwykłe `cp -r` albo `rsync -av`)

**Ważne zastrzeżenia:**
- Skopiuj **cały folder `models/`** (i `manifests/`, i `blobs/`) — same manifesty bez blobów są bezużyteczne i odwrotnie.
- Wersje Ollamy na obu komputerach powinny być zbliżone (`ollama --version`) — format magazynu rzadko się zmienia, ale przy dużej rozbieżności wersji może się nie zgrać.
- Po skopiowaniu zawsze sprawdź: `ollama list` — powinny pokazać się wszystkie modele bez ponownego pobierania.
- Jeśli `ollama list` nie widzi modeli mimo skopiowania plików: zrestartuj usługę Ollamy (Windows: ikona w zasobniku → Restart / `taskkill /IM ollama.exe /F` i uruchom ponownie; Linux: `sudo systemctl restart ollama`).

---

## 🦙 Ściągawka — komendy Ollama

| Komenda | Co robi |
|---|---|
| `ollama list` | lista pobranych modeli lokalnie |
| `ollama pull <model>` | pobierz/zaktualizuj model (np. `ollama pull llava`) |
| `ollama rm <model>` | usuń model, zwolnij miejsce |
| `ollama show <model>` | szczegóły modelu (parametry, template, license) |
| `ollama show <model> --modelfile` | pokaż pełny Modelfile (system prompt, parametry domyślne) |
| `ollama ps` | modele aktualnie załadowane w pamięci (aktywne) |
| `ollama stop <model>` | wyładuj model z pamięci (zwolnij RAM/VRAM bez usuwania pliku) |
| `ollama cp <model> <nowa-nazwa>` | skopiuj model pod nową nazwą (np. do własnego wariantu) |
| `ollama run <model>` | czat interaktywny w terminalu (do szybkiego testu bez naszej appki) |
| `ollama --version` | sprawdź wersję (ważne przy przenoszeniu modeli między maszynami) |

**Filtrowanie modeli wizyjnych:** nie każdy model obsługuje obrazy — jeśli nazwa nie ma w sobie `vision`/`vl`/`llava`/`moondream`/`minicpm`, prawdopodobnie to model czysto tekstowy. Taki przyjmie zapytanie z obrazkiem bez błędu, ale zwróci pustą odpowiedź (appka nie zgłosi błędu, po prostu nic nie wygeneruje).

---

## 🤖 Dodawanie modeli do interfejsu

Po `ollama pull <model>`, dopisz wiersz do `llm_models/ollama_models.csv`:
```csv
model_id,model_name,size,description
5,llava:13b,7.4GB,Wieksza wersja llava - lepsza jakosc, wolniej
```

**Uwaga na przecinki w opisie** — jeśli opis zawiera przecinek, otocz go cudzysłowami:
```csv
5,llava:13b,7.4GB,"Wieksza wersja, lepsza jakosc"
```
(bez cudzysłowów CSV się rozjedzie i całość przestanie się wczytywać)

Modele HuggingFace analogicznie w `llm_models/hf_models.csv` — pobierają się automatycznie przy pierwszym użyciu (potrzebny internet przy pierwszym razie, potem z cache).

---

## 🌐 REST API — endpointy

Pełna dokumentacja (Swagger): `http://localhost:8000/docs`

- `GET /models?backend=ollama|hf` — lista modeli z CSV, wzbogacona o `capabilities` (dla Ollamy pobrane z `/api/show`, cache'owane w pamieci; dla HF na sztywno `["vision"]`)
- `POST /models/refresh-capabilities` — czysci cache capabilities (przydatne po `ollama pull` nowego modelu bez restartu API)
- `POST /generate` — generowanie opisu (`backend`, `model_id`, `system_prompt`, `user_prompt`, `temperature`, `top_p`, `max_tokens`, `ctx_len` — tylko Ollama, `images` — jeden lub wiele plików, przetwarzane sekwencyjnie). Dla Ollamy odrzuca zapytanie wczesniej (bez wywolywania modelu), jesli model nie ma capability `vision`.
- `GET /raport` — historia z `raports/raport.csv`

**Wykrywanie capabilities (Ollama):** UI pokazuje przy kazdym modelu skrot capabilities w nawiasach kwadratowych (np. `[VC]` = vision + completion, `[CT]` = completion + tools, `[CTh]` = completion + thinking). Jesli wybierzesz model bez `vision` i podepniesz zdjecie, appka poprosi o potwierdzenie zanim wyśle zapytanie (bo najprawdopodobniej zwroci blad/pusta odpowiedz).

**Wykrywanie RAM/VRAM + flagowanie ryzyka (`api/services/system_service.py`):**
- `GET /system/resources` — aktualny wolny/calkowity RAM (przez `psutil`) i VRAM (przez `nvidia-smi`, jesli GPU to NVIDIA; brak `nvidia-smi` w PATH = zakladamy inference na CPU, VRAM=0). AMD/Intel GPU nie sa jeszcze obslugiwane.
- `/models` dokłada pole `hardware: {status, powod}` per model. Rozmiar modelu brany jest w pierwszej kolejnosci z `client.list()` Ollamy (dokladny, w bajtach), a jesli model nie jest jeszcze pobrany — parsowany z kolumny `size` w CSV (np. `"5.5GB"`).
- Statusy: `bezpieczny` (miesci sie w wolnym VRAM z 20% zapasu), `ryzykowny` (miesci sie w RAM+VRAM lacznie, ale czesc pojdzie na CPU — wolniej), `za_duzy` (przekracza dostepna pamiec), `nieznany` (nie udalo sie ustalic rozmiaru).
- UI pokazuje pasek `RAM: x/y GB wolne · VRAM: x/y GB wolne` w headerze, symbol statusu (`✓`/`⚠`/`✗`/`?`) przy kazdym modelu w dropdownie, oraz pelny opis pod selectem.
- `POST /models/refresh-capabilities` czysci teraz rowniez cache rozmiarow (nie tylko capabilities).

## 🔧 Częste problemy

- **`pip` nie działa mimo aktywnego venv:** użyj `python -m pip ...`
- **`Cannot allocate memory` przy większym modelu:** za mało RAM na model + system jednocześnie — spróbuj mniejszego modelu (`moondream`, `llava:7b`) albo `ollama stop <inny-model>`, żeby zwolnić pamięć przed uruchomieniem kolejnego.
- **Model zwraca pusty opis:** model prawdopodobnie nie obsługuje obrazów (patrz ściągawka wyżej) — appka ma wbudowany fallback tylko na problem z rolą `system`, nie na brak wsparcia dla obrazów w ogóle.
- **Historia się nie ładuje:** stary `raport.csv` z inną strukturą kolumn — aplikacja archiwizuje go automatycznie przy następnym zapisie.
- **Port 8000 zajęty:** sprawdź czy nie masz już uruchomionego innego `uvicorn` (np. z poprzedniej sesji terminala) — `Ctrl+C` w tamtym oknie albo zmień port: `--port 8001`.
