# Dokumentacja projektu ASI — Analiza wypadków drogowych

Dokumentacja systemu predykcji stopnia obrażeń w wypadkach drogowych.

## Spis treści

1. [SETUP.md](SETUP.md) — instalacja i uruchomienie
2. [ARCHITECTURE.md](ARCHITECTURE.md) — architektura i podział na warstwy
3. [API.md](API.md) — dokumentacja API FastAPI
4. [DATA_DICTIONARY.md](DATA_DICTIONARY.md) — słownik danych
5. [MODELS.md](MODELS.md) — opis modeli i wyników
6. [CONTRIBUTING.md](CONTRIBUTING.md) — jak dodać nowy pipeline

## Szybki start

API w Dockerze:
```bash
docker-compose up --build
# Swagger: http://localhost:8000/docs
```

Pełny projekt:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
kedro run
uvicorn crash_kedro.api.app:app --reload
```

Więcej w [SETUP.md](SETUP.md).

## Autorzy

Projekt zaliczeniowy z przedmiotu ASI na PJATK.

- Artur Cichocki
- Bartosz Pikutin
- Wiktor Golba
