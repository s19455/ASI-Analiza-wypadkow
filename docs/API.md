# Dokumentacja API FastAPI

## Przegląd

API zapewnia predykcje stopnia obrażeń w wypadkach drogowych poprzez REST.

**URL:** `http://localhost:8000` (dev) lub `http://api.production.com` (prod)

**Dokumentacja interaktywna:**
- Swagger UI: `/docs`
- ReDoc: `/redoc`

---

## Endpointy

### 1. `POST /predict` — Predykcja dla pojedynczego zdarzenia

Wykonuje predykcję stopnia obrażeń dla jednego rekordu wypadku.

**Request:**
```http
POST /predict HTTP/1.1
Content-Type: application/json

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
  "vehicle_movement": "MOVING CONSTANT",
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
  "probabilities": {
    "NO_INJURY": 0.82,
    "MINOR": 0.17,
    "SERIOUS": 0.01
  },
  "timestamp": "2026-05-23T14:30:45.123456"
}
```

**Parametry wejściowe:**

| Pole | Typ | Domyślnie | Opis |
|------|-----|----------|------|
| `weather` | string | CLEAR | Pogoda (CLEAR, CLOUDY, RAIN, FOG, itd.) |
| `light` | string | DAYLIGHT | Warunki oświetlenia (DAYLIGHT, DARK itd.) |
| `collision_type` | string | SAME DIR REAR END | Typ kolizji |
| `surface_condition` | string | DRY | Stan powierzchni (DRY, WET, SNOW, itd.) |
| `traffic_control` | string | NO CONTROLS | Kontrola ruchu (TRAFFIC SIGNAL, STOP SIGN, itd.) |
| `driver_substance_abuse` | string | NONE DETECTED | Wpływ alkoholu/narkotyków |
| `driver_distracted_by` | string | NOT DISTRACTED | Rozproszenie uwagi kierowcy |
| `vehicle_body_type` | string | PASSENGER CAR | Typ pojazdu (MOTORCYCLE, TRUCK, itd.) |
| `vehicle_damage_extent` | string | FUNCTIONAL | Zakres szkód (MINIMAL, MODERATE, SEVERE, itd.) |
| `vehicle_movement` | string | MOVING CONSTANT | Ruch pojazdu (STOPPED, MOVING, itd.) |
| `speed_limit` | integer | 35 | Ograniczenie prędkości (km/h) **0-120** |
| `driver_at_fault` | string | Yes | Kierowca winny (Yes/No) |
| `driverless_vehicle` | string | No | Pojazd autonomiczny (Yes/No) |
| `parked_vehicle` | string | No | Pojazd zaparkowany (Yes/No) |
| `vehicle_year` | integer | 2020 | Rok produkcji pojazdu (1980-2026) |
| `crash_hour` | integer | 12 | Godzina wypadku (0-23) |
| `crash_dayofweek` | integer | 2 | Dzień tygodnia (0=poniedziałek, 6=niedziela) |
| `crash_month` | integer | 6 | Miesiąc (1-12) |
| `crash_year` | integer | 2026 | Rok wypadku |

**Odpowiedź:**

| Pole | Typ | Opis |
|------|-----|------|
| `severity` | string | Przewidywana klasa (NO_INJURY, MINOR, SERIOUS) lub UNKNOWN |
| `probabilities` | object | Prawdopodobieństwa dla każdej klasy (0.0-1.0) |
| `timestamp` | string | ISO 8601 timestamp żądania |

**Kody błędów:**
- `200 OK` — sukces
- `422 Unprocessable Entity` — błąd walidacji wejścia
- `500 Internal Server Error` — błąd serwera (np. model nie załadowany)

---

### 2. `GET /health` — Status zdrowia aplikacji

Sprawdza, czy aplikacja i model są gotowe do pracy.

**Request:**
```http
GET /health HTTP/1.1
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "encoders_loaded": true,
  "model_path": "/app/data/06_models/tuned_model.pkl"
}
```

**Odpowiedź:**

| Pole | Typ | Opis |
|------|-----|------|
| `status` | string | "ok" (model załadowany) lub "no model loaded" |
| `model_loaded` | boolean | Czy model jest załadowany |
| `encoders_loaded` | boolean | Czy encodery są załadowane |
| `model_path` | string \| null | Ścieżka do załadowanego modelu (lub null) |

**Zastosowanie:**
- Sprawdzanie gotowości serwisu (Kubernetes readiness probe)
- Monitorowanie (health checks co X sekund)
- Debugging

---

### 3. `GET /predictions/recent` — Ostatnie predykcje

Zwraca ostatnie predykcje z logów (JSONL).

**Request:**
```http
GET /predictions/recent?n=10 HTTP/1.1
```

**Query Parametry:**

| Parametr | Typ | Default | Opis |
|----------|-----|---------|------|
| `n` | integer | 10 | Liczba ostatnich predykcji |

**Response (200 OK):**
```json
{
  "predictions": [
    {
      "timestamp": "2026-05-23T14:30:45.123456",
      "input": {
        "weather": "CLEAR",
        "light": "DAYLIGHT",
        ...
      },
      "prediction": "NO_INJURY",
      "probabilities": {
        "NO_INJURY": 0.82,
        "MINOR": 0.17,
        "SERIOUS": 0.01
      }
    },
    ...
  ]
}
```

**Użyteczność:**
- Monitoring predykcji w real-time
- Debugowanie
- Analiza wzorców zapytań

---

## Kod HTTP

| Kod | Znaczenie | Kiedy |
|-----|-----------|-------|
| 200 | OK | Sukces |
| 422 | Unprocessable Entity | Błędne dane wejściowe (widać w /docs) |
| 500 | Internal Server Error | Błąd w przetwarzaniu (serwer) |

---

## Przykłady użycia

### cURL

#### Predykcja
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "weather": "CLEAR",
    "light": "DAYLIGHT",
    "collision_type": "SAME DIR REAR END",
    "surface_condition": "DRY",
    "traffic_control": "NO CONTROLS",
    "driver_substance_abuse": "NONE DETECTED",
    "driver_distracted_by": "NOT DISTRACTED",
    "vehicle_body_type": "PASSENGER CAR",
    "vehicle_damage_extent": "FUNCTIONAL",
    "vehicle_movement": "MOVING CONSTANT",
    "speed_limit": 35,
    "driver_at_fault": "Yes",
    "driverless_vehicle": "No",
    "parked_vehicle": "No",
    "vehicle_year": 2020,
    "crash_hour": 12,
    "crash_dayofweek": 2,
    "crash_month": 6,
    "crash_year": 2026
  }'
```

#### Health check
```bash
curl http://localhost:8000/health
```

#### Ostatnie predykcje
```bash
curl "http://localhost:8000/predictions/recent?n=5"
```

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Predykcja
payload = {
    "weather": "CLEAR",
    "light": "DAYLIGHT",
    "collision_type": "SAME DIR REAR END",
    "surface_condition": "DRY",
    "traffic_control": "NO CONTROLS",
    "driver_substance_abuse": "NONE DETECTED",
    "driver_distracted_by": "NOT DISTRACTED",
    "vehicle_body_type": "PASSENGER CAR",
    "vehicle_damage_extent": "FUNCTIONAL",
    "vehicle_movement": "MOVING CONSTANT",
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

response = requests.post(f"{BASE_URL}/predict", json=payload)
result = response.json()
print(f"Severity: {result['severity']}")
print(f"Probabilities: {result['probabilities']}")
```

### JavaScript/TypeScript

```typescript
const BASE_URL = "http://localhost:8000";

async function predict(data: object) {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  return response.json();
}

const payload = {
  weather: "CLEAR",
  light: "DAYLIGHT",
  // ... pozostałe pola
};

const result = await predict(payload);
console.log(`Severity: ${result.severity}`);
```

---

## Błędy i debugowanie

### Błąd 422 — Unprocessable Entity

Zwykle znaczy, że pole ma złą wartość lub jest puste.

**Przykład:**
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "speed_limit"],
      "msg": "ensure this value is greater than or equal",
      "input": -5
    }
  ]
}
```

**Rozwiązanie:**
- Sprawdź typy danych (integer vs string)
- Sprawdź zakresy wartości
- Użyj `/docs` do interaktywnego testowania

### Błąd 500 — Internal Server Error

Zwykle znaczy, że model nie jest załadowany lub serwer ma problem.

**Rozwiązanie:**
- Sprawdź `/health`
- Sprawdź logi serwera
- Upewnij się, że `data/06_models/model.pkl` istnieje

---

## Limitacje i znane problemy

1. **Model nie obsługuje nowych kategorii** — unknown kategorie kodowane jako `-1`
2. **Brak autentykacji** — API jest publiczne, w produkcji dodaj JWT/API keys
3. **Logowanie predykcji jest synchrne** — może spowolnić żądanie przy błędzie dysku
4. **Brak rate limitingu** — w produkcji dodaj Redis/slowapi

---

## Rozszerzenia

### Autentykacja (FastAPI + security)
```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/predict")
def predict(input_data: CrashInput, credentials = Depends(security)):
    # Verify JWT token
    ...
```

### Rate limiting
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
def predict(...):
    ...
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model():
    ...
```

---

## SLA i performans

| Metrika | Cel | Aktualne |
|---------|-----|----------|
| Latency (predykcja) | <100ms | ~50ms |
| Dostępność | 99.9% | ~100% (dev) |
| Max RPS | 100 | ~50 (single instance) |
| Max payload | 1 MB | ~1 KB (input) |

---

## Integracja z systemami

### Webhook do innego serwisu
```python
async def predict_and_notify(input_data: CrashInput):
    result = predict_crash_severity(input_data)
    await notify_external_service(result)
    return result
```

### Message Queue (Celery)
```python
from celery import Celery

@celery_app.task
def async_predict(data):
    return predict_crash_severity(CrashInput(**data))
```

---

## Podsumowanie

API jest proste, ale potężne:
- **Łatwe do użycia** — REST + JSON
- **Dobrze udokumentowane** — Swagger UI
- **Rozszerzalne** — FastAPI ułatwia dodawanie funkcji
- **Skalowalne** — gotowe do uruchomienia w Dockerze/Kubernetesie

