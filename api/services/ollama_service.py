import os
import ollama

# W Dockerze OLLAMA_HOST ustawia docker-compose.yml (wskazuje na "ollama-serwer").
# Natywnie (Windows/Linux/macOS bez Dockera) leci na domyslny localhost.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_HOST)

# Cache capabilities modeli, zeby nie odpytywac Ollamy (/api/show) przy kazdym
# odswiezeniu listy modeli w UI - capabilities modelu nie zmieniaja sie w czasie
# dzialania aplikacji (chyba ze ktos podmieni model tym samym tagiem).
_capabilities_cache: dict = {}


def _wyciagnij_capabilities(info) -> list:
    """`ollama.Client.show()` zwraca obiekt ShowResponse (nowsze wersje biblioteki)
    albo zwykly dict (starsze wersje) - obslugujemy oba warianty defensywnie."""
    capabilities = getattr(info, "capabilities", None)
    if capabilities is None and isinstance(info, dict):
        capabilities = info.get("capabilities")
    return list(capabilities or [])


def get_capabilities(model_name: str, refresh: bool = False) -> list:
    """Zwraca liste capabilities modelu z Ollamy, np. ['completion', 'vision', 'tools', 'thinking'].
    Wymaga, zeby model byl juz pobrany (`ollama pull`) - dla nieistniejacego modelu zwraca [].
    """
    if not refresh and model_name in _capabilities_cache:
        return _capabilities_cache[model_name]
    try:
        info = client.show(model_name)
    except Exception:
        # Model nieznany Ollamie (np. literowka w CSV, albo jeszcze nie pobrany) -
        # nie wywalamy calego /models, po prostu brak informacji o capabilities.
        return []
    capabilities = _wyciagnij_capabilities(info)
    _capabilities_cache[model_name] = capabilities
    return capabilities


def clear_capabilities_cache():
    _capabilities_cache.clear()


def _wywolaj(model_name, messages, temperature, top_p, max_tokens, ctx_len=4096):
    try:
        response = client.chat(
            model=model_name,
            messages=messages,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "num_ctx": ctx_len
            },
        )

        # Bezpieczne pobieranie: jeśli get() zwróci None, zastąp pustym ciągiem
        opis = response.get("message", {}).get("content") or ""

        # Dodatkowy warunek dla modeli typu "thinking"
        if not opis.strip():
            opis = response.get("message", {}).get("reasoning_content") or ""

        # Ostateczne zabezpieczenie - jeśli nadal jest puste/None
        if not opis:
            return "Błąd: Model nie zwrócił żadnego tekstu."

        return opis

    except Exception as e:
        print(f"Ollama Call Error: {e}")
        return f"Błąd komunikacji z Ollamą: {str(e)}"


def generate(
    model_name: str,
    image_path: str,
    system_prompt: str = "",
    user_prompt: str = "Opisz szczegolowo, co znajduje sie na tym zdjeciu.",
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 200,
    ctx_len: int = 4096,
) -> str:
    capabilities = get_capabilities(model_name)
    if capabilities and "vision" not in capabilities:
        raise ValueError(
            f"Model '{model_name}' nie ma capability 'vision' (ma: {capabilities or 'brak danych'}) "
            f"- nie obsluguje obrazow, zapytanie pominieto zamiast wysylac je na sile."
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": user_prompt,
        "images": [image_path],
    })

    opis = _wywolaj(model_name, messages, temperature, top_p, max_tokens, ctx_len)

    # Niektore mniejsze/prostsze modele (np. moondream) nie obsluguja dobrze
    # oddzielnej roli "system" i po cichu zwracaja pusta odpowiedz (bez bledu).
    # W takim przypadku ponawiamy, laczac prompt systemowy z promptem
    # uzytkownika w jednej wiadomosci "user".
    if not opis.strip() and system_prompt:
        polaczony_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        fallback_messages = [{
            "role": "user",
            "content": polaczony_prompt,
            "images": [image_path],
        }]
        opis = _wywolaj(model_name, fallback_messages, temperature, top_p, max_tokens, ctx_len)

    return opis
