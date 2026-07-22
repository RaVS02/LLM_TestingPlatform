from transformers import pipeline

# Cache zaladowanych pipeline'ow, zeby nie ladowac wag modelu od nowa
# przy kazdym zapytaniu API (to najdrozsza operacja).
_pipelines_cache = {}


def _wybierz_urzadzenie():
    """Zwraca numer urzadzenia dla transformers.pipeline: 0 = pierwsze GPU, -1 = CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return -1


def _get_pipeline(model_path: str):
    if model_path not in _pipelines_cache:
        device = _wybierz_urzadzenie()
        # UWAGA: nowsze wersje transformers usunely task "image-to-text" na rzecz
        # "image-text-to-text", ktory dziala na formacie wiadomosci czatu (jak modele
        # multimodalne typu LLaVA) zamiast prostego (prompt, obraz).
        _pipelines_cache[model_path] = pipeline("image-text-to-text", model=model_path, device=device)
    return _pipelines_cache[model_path]


def generate(
    model_path: str,
    image_path: str,
    system_prompt: str = "",
    user_prompt: str = "Opisz szczegolowo, co znajduje sie na tym zdjeciu.",
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 200,
) -> str:
    pipe = _get_pipeline(model_path)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": user_prompt},
        ],
    })

    generate_kwargs = {"max_new_tokens": max_tokens}
    if temperature and temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    outputs = pipe(text=messages, return_full_text=False, **generate_kwargs)
    return outputs[0]["generated_text"]
