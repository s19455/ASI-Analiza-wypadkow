# Dokumentacja projektu ASI — Analiza wypadków drogowych

Dokumentacja techniczna, architekturalna i użytkownika dla systemu predykcji stopnia obrażeń w wypadkach drogowych.

## Spis treści

1. **[SETUP.md](SETUP.md)** — instrukcje instalacji i uruchomienia
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** — opis architektury i modularności
3. **[API.md](API.md)** — dokumentacja API FastAPI
4. **[DATA_DICTIONARY.md](DATA_DICTIONARY.md)** — słownik danych wejściowych
5. **[MODELS.md](MODELS.md)** — opis modeli i wyników
6. **[CONTRIBUTING.md](CONTRIBUTING.md)** — wytyczne dla deweloperów

## Szybki start

### Uruchomienie API w Dockerze
```bash
docker-compose up --build
# API będzie dostępne pod http://localhost:8000
# Swagger under http://localhost:8000/docs
```

### Uruchomienie pełnego projektu
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
kedro run
uvicorn crash_kedro.api.app:app --reload
```

Więcej szczegółów w [SETUP.md](SETUP.md).

## Struktura dokumentacji

```
docs/
├── README.md              # Ten plik
├── SETUP.md              # Instrukcje instalacji
├── ARCHITECTURE.md       # Opis architektury
├── API.md                # Dokumentacja API
├── DATA_DICTIONARY.md    # Słownik danych
├── MODELS.md             # Opis modeli
└── CONTRIBUTING.md       # Wytyczne dla deweloperów
```

## Informacje kontaktowe

Projekt zaliczeniowy z przedmiotu **Środowiska uruchomieniowe Machine Learning (ASI)** na PJATK.

Autorzy:
- Artur Cichocki
- Bartosz Pikutin
- Wiktor Golba

Data: maj 2026

