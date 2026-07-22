import os
from datetime import datetime
import pandas as pd


def load_models(sciezka):
    """
    Wczytuje liste modeli z pliku CSV, bez drukowania (do uzycia w API).
    Dziala dla roznych schematow CSV (np. Ollama: model_id,model_name,size,description
    HF: model_id,model_name,path,size,description).
    """
    if not os.path.isfile(sciezka):
        raise FileNotFoundError(
            f"Nie znaleziono pliku z konfiguracja modeli: {sciezka}\n"
            f"Sprawdz, czy plik istnieje w folderze llm_models."
        )
    return pd.read_csv(sciezka)


def print_list_models(sciezka):
    """Wczytuje i wyswietla liste modeli z pliku CSV (uzywane w trybie konsolowym)."""
    modele = load_models(sciezka)

    if modele.empty:
        print(f"\nPlik {sciezka} jest pusty - brak zdefiniowanych modeli.")
        return modele

    print("\nDostepne modele:")
    print(modele.to_string(index=False))
    return modele


def create_raport(
    image: str,
    nazwa_modelu: str,
    opis: str,
    system_prompt: str = "",
    user_prompt: str = "",
    temperature=None,
    top_p=None,
    max_tokens=None,
):
    df = pd.DataFrame({
        'nazwa_pliku_obrazu': [image],
        'nazwa_modelu': [nazwa_modelu],
        'prompt_systemowy': [system_prompt],
        'prompt_uzytkownika': [user_prompt],
        'temperatura': [temperature],
        'top_p': [top_p],
        'max_tokeny': [max_tokens],
        'wygenerowany_opis': [opis],
    })

    path_to_save = "./raports"
    os.makedirs(path_to_save, exist_ok=True)
    full_path = os.path.join(path_to_save, 'raport.csv')

    try:
        file_exists = os.path.isfile(full_path)

        if file_exists:
            # Jesli istniejacy plik ma inna strukture kolumn (np. zostal utworzony
            # przez wczesniejsza wersje aplikacji), archiwizujemy go zamiast mieszac
            # niekompatybilne naglowki w jednym pliku CSV.
            istniejacy_naglowek = pd.read_csv(full_path, nrows=0).columns.tolist()
            if istniejacy_naglowek != list(df.columns):
                backup_path = os.path.join(
                    path_to_save,
                    f"raport_archiwum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                )
                os.rename(full_path, backup_path)
                file_exists = False

        df.to_csv(full_path, mode='a', header=not file_exists, index=False)
    except Exception as e:
        return e
    return True
