# Dokumentacja API FastAPI

API serwuje predykcje stopnia obrażeń w wypadkach drogowych przez REST.

**URL (lokalnie):** `http://localhost:8000`
**Dokumentacja interaktywna:** Swagger UI pod `/docs`, ReDoc pod `/redoc`

## Endpointy

### `POST /predict` — predykcja dla pojedynczego zdarzenia

**Request:**
```json
{
  "weather": "CLEAR",
  "light": "DAYLIGHT",
  "collision_type": "SAME DIR REAR END",
  "surface_condition": "DRY",
  "traffic_control": "NO CONTROLS",
  "driver_substance_abuse": "NONE DETECTED",
  "driver_distracted_by": "NOT DISTRACTED",
  "vehicle_body_type": "PASSENGER CAR",
  "vehicle_damage_extent": "FUNCTIONAL",
  "vehicle_movement": "MOVING CONSTANT SPEED",
  "speed_limit": 35,
  "driver_at_fault": "Yes",
  "driverless_vehicle": "No",
  "parked_vehicle": "No",
  "vehicle_year": 2020,
  "crash_hour": 12,
  "crash_dayofweek": 2,
  "crash_month": 6,
  "crash_year": 2026
}
```

**Response (200 OK):**
```json
{
  "severity": "NO_INJURY",
  "probabilities": { "NO_INJURY": 0.82, "MINOR": 0.17, "SERIOUS": 0.01 },
  "timestamp": "2026-05-23T14:30:45.123456"
}
```

**Parametry wejściowe:**

| Pole | Typ | Domyślnie | Opis |
|------|-----|----------|------|
| `weather` | string | CLEAR | Pogoda (CLEAR, CLOUDY, RAINING, FOGGY, SNOW, ...) |
| `light` | string | DAYLIGHT | Oświetlenie (DAYLIGHT, DARK LIGHTS ON, DARK NO LIGHTS, ...) |
| `collision_type` | string | SAME DIR REAR END | Typ kolizji |
| `surface_condition` | string | DRY | Stan nawierzchni (DRY, WET, SNOW, ICE, ...) |
| `traffic_control` | string | NO CONTROLS | Kontrola ruchu (TRAFFIC SIGNAL, STOP SIGN, ...) |
| `driver_substance_abuse` | string | NONE DETECTED | Wpływ alkoholu/narkotyków |
| `driver_distracted_by` | string | NOT DISTRACTED | Rozproszenie uwagi kierowcy |
| `vehicle_body_type` | string | PASSENGER CAR | Typ pojazdu |
| `vehicle_damage_extent` | string | FUNCTIONAL | Zakres uszkodzeń |
| `vehicle_movement` | string | MOVING CONSTANT SPEED | Ruch pojazdu |
| `speed_limit` | integer | 35 | Ograniczenie prędkości |
| `driver_at_fault` | string | Yes | Kierowca winny (Yes/No) |
| `driverless_vehicle` | string | No | Pojazd autonomiczny (Yes/No) |
| `parked_vehicle` | string | No | Pojazd zaparkowany (Yes/No) |
| `vehicle_year` | integer | 2020 | Rok produkcji pojazdu |
| `crash_hour` | integer | 12 | Godzina wypadku (0-23) |
| `crash_dayofweek` | integer | 2 | Dzień tygodnia (0=poniedziałek, 6=niedziela) |
| `crash_month` | integer | 6 | Miesiąc (1-12) |
| `crash_year` | integer | bieżący rok | Rok wypadku |

Wartości tekstowe powinny pochodzić ze słownika danych treningowych — nieznane są kodowane jako `-1`. Listy dostępnych wartości podpowiada frontend Streamlit (czyta je z `encoders.pkl`).

**Odpowiedź:**

| Pole | Typ | Opis |
|------|-----|------|
| `severity` | string | Klasa: NO_INJURY, MINOR, SERIOUS lub UNKNOWN |
| `probabilities` | object | Prawdopodobieństwa klas (0.0-1.0) |
| `timestamp` | string | Czas predykcji (ISO 8601) |

### `GET /health` — status aplikacji

```json
{
  "status": "ok",
  "model_loaded": true,
  "encoders_loaded": true,
  "model_path": "/app/data/06_models/model.pkl"
}
```

| Pole | Opis |
|------|------|
| `status` | "ok" gdy model załadowany, inaczej "no model loaded" |
| `model_loaded` | czy model jest załadowany |
| `encoders_loaded` | czy enkodery są załadowane |
| `model_path` | ścieżka do modelu lub null |

### `GET /predictions/recent` — ostatnie predykcje

Zwraca ostatnie predykcje z logu `logs/predictions.jsonl`.

**Request:** `GET /predictions/recent?n=10`

| Parametr | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `n` | integer | 10 | Liczba ostatnich predykcji |

```json
{
  "predictions": [
    {
      "timestamp": "2026-05-23T14:30:45.123456",
      "input": { "weather": "CLEAR", "...": "..." },
      "prediction": "NO_INJURY",
      "probabilities": { "NO_INJURY": 0.82, "MINOR": 0.17, "SERIOUS": 0.01 }
    }
  ]
}
```

## Kody odpowiedzi

| Kod | Znaczenie |
|-----|-----------|
| 200 | OK |
| 422 | Błędne dane wejściowe (np. nieznane pole — model używa `extra=forbid`) |
| 500 | Błąd serwera |

## Przykłady

cURL:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"weather": "CLEAR", "speed_limit": 35}'

curl http://localhost:8000/health
curl "http://localhost:8000/predictions/recent?n=5"
```

Python:
```python
import requests

payload = {"weather": "CLEAR", "speed_limit": 35}
r = requests.post("http://localhost:8000/predict", json=payload)
print(r.json()["severity"])
```

## Znane ograniczenia

- Model nie obsługuje nowych kategorii — nieznane wartości są kodowane jako `-1`.
- Brak autentykacji i rate limitingu — API jest przeznaczone do uruchomienia lokalnego / na potrzeby projektu.
- Logowanie predykcji jest synchroniczne (zapis do pliku JSONL przy każdym żądaniu).
