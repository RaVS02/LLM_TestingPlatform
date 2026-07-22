import os
import ollama

# W Dockerze OLLAMA_HOST ustawia docker-compose.yml (wskazuje na "ollama-serwer").
# Natywnie (Windows/Linux/macOS bez Dockera) leci na domyslny localhost.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_HOST)


def _wywolaj(model_name, messages, temperature, top_p, max_tokens, ctx_len):
    response = client.chat(
        model=model_name,
        messages=messages,
        options={
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "num_ctx": ctx_len,
        },
    )
    return response["message"]["content"]


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
