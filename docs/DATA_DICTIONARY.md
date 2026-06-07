# Słownik danych (Data Dictionary)

## Przegląd

Ten dokument opisuje wszystkie zmienne w zbiorze danych o wypadkach drogowych z Montgomery County, Maryland (2015-2024).

---

## Dane wejściowe (Raw)

**Źródło:** Montgomery County Police Department, Maryland Open Data  
**Format:** CSV  
**Ścieżka:** `data/01_raw/crash_data.csv`  
**Rozmiar:** ~172,000 wierszy × 43 kolumny

---

## Zmienne w surowych danych

### Informacje o wypadku

| Kolumna | Typ | Opis | Przykłady |
|---------|-----|------|----------|
| Report Number | string | Unikalny identyfikator raportu | RB123456 |
| Local Case Number | string | Lokalny numer sprawy | LCN789 |
| Crash Date/Time | datetime | Data i czas wypadku | 2024-05-15 14:30:00 |
| Agency Name | string | Agencja raportująca | Montgomery Police |
| Location | string | Opis lokalizacji | On the roadway |

### Warunkami

| Kolumna | Typ | Opis | Przykłady |
|---------|-----|------|----------|
| Weather | string | Warunki pogodowe | CLEAR, CLOUDY, RAIN, FOG, SNOW |
| Light | string | Warunki oświetlenia | DAYLIGHT, DARK |
| Surface Condition | string | Stan powierzchni jezdni | DRY, WET, SNOW |
| Traffic Control | string | Rodzaj kontroli ruchu | NO CONTROLS, TRAFFIC SIGNAL, STOP SIGN |

### Strona kierowcy

| Kolumna | Typ | Opis | Przykłady |
|---------|-----|------|----------|
| Driver Substance Abuse | string | Wpływ alkoholu/narkotyków | NONE DETECTED, IMPAIRED BY ALCOHOL, ... |
| Driver At Fault | string | Czy kierowca jest winny | Yes, No |
| Driver Distracted By | string | Rodzaj rozproszenia uwagi | NOT DISTRACTED, CELL PHONE, ... |
| Drivers License State | string | Stan wydania prawa jazdy | MD, VA, NC, ... |

### Pojazd

| Kolumna | Typ | Opis | Przykłady |
|---------|-----|------|----------|
| Vehicle Year | integer | Rok produkcji | 2015, 2020, 2025 |
| Vehicle Make | string | Producent | TOYOTA, FORD, BMW |
| Vehicle Model | string | Model | CIVIC, FOCUS, X5 |
| Vehicle Body Type | string | Typ karoserii | PASSENGER CAR, MOTORCYCLE, TRUCK |
| Vehicle Movement | string | Typ ruchu pojazdu | MOVING CONSTANT, STOPPED, ... |
| Equipment Problems | string | Problemy techniczne | None, BRAKE FAILURE, ... |

### Wektysja i obrażenia

| Kolumna | Tip | Opis | Przykłady |
|---------|-----|------|----------|
| **Injury Severity** (TARGET) | string | Poziom obrażeń | NO APPARENT INJURY, POSSIBLE INJURY, SUSPECTED MINOR INJURY, SUSPECTED SERIOUS INJURY, FATAL INJURY |
| Non-Motorist | string | Typ pieszego/rowerzysty | None, ADULT PEDESTRIAN, PEDALCYCLIST |

### Inne

| Kolumna | Typ | Opis |
|---------|-----|------|
| Collision Type | string | Typ kolizji (np. SAME DIR REAR END) |
| Circumstance | string | Opisowa okoliczność wypadku |
| Latitude | float | Współrzędna geograficzna |
| Longitude | float | Współrzędna geograficzna |

---

## Dane po przetworzeniu (Primary)

**Ścieżka:** `data/03_primary/crash_features.parquet`

Po przetworzeniu w pipeline'ie `data_preparation`:

### Zmienne engineeryjne (cechy)

| Kolumna | Typ | Opis | Zakres |
|---------|-----|------|--------|
| crash_hour | integer | Godzina wypadku | 0-23 |
| crash_dayofweek | integer | Dzień tygodnia | 0 (pn) - 6 (nd) |
| crash_month | integer | Miesiąc | 1-12 |
| crash_year | integer | Rok | 2015-2024 |
| is_night | integer | Czy noc (DARK) | 0, 1 |
| is_bad_weather | integer | Zła pogoda (nie CLEAR/CLOUDY) | 0, 1 |
| is_wet_surface | integer | Mokra równiła (nie DRY) | 0, 1 |
| vehicle_age | integer | Wiek pojazdu w latach | 0-50 |

### Zmienne kategoryczne (enkodowane)

Wszystkie zmienne `object` mają: **0, 1, 2, ..., N** (gdzie N = liczba unikalnych wartości)

**Przykład:**
```
Weather: CLEAR → 0, CLOUDY → 1, RAIN → 2, FOG → 3, ...
```

Nieznane wartości (unseen during training): **-1**

### Zmienna docelowa

| Kolumna | Typ | Opis | Klasy |
|---------|-----|------|-------|
| **Severity_Group** (TARGET) | string | Zmapowana klasa obrażeń | NO_INJURY, MINOR, SERIOUS |

**Mapowanie:**
```yaml
NO_INJURY:
  - NO APPARENT INJURY

MINOR:
  - POSSIBLE INJURY
  - SUSPECTED MINOR INJURY

SERIOUS:
  - SUSPECTED SERIOUS INJURY
  - FATAL INJURY
```

**Rozkład klas:**
```
NO_INJURY:  ~82% (imbalanced)
MINOR:      ~17%
SERIOUS:    ~1%
```

---

## Zakresy parametrów API

Przy wysyłaniu żądania `/predict`, należy używać:

| Parametr | Typ | Zakres/Wartości | Domyślnie |
|----------|-----|-----------------|----------|
| weather | string | CLEAR, CLOUDY, RAIN, FOG, SNOW, ... | CLEAR |
| light | string | DAYLIGHT, DARK, ... | DAYLIGHT |
| collision_type | string | Dowolna z danych treningowych | SAME DIR REAR END |
| surface_condition | string | DRY, WET, SNOW, ... | DRY |
| traffic_control | string | NO CONTROLS, TRAFFIC SIGNAL, ... | NO CONTROLS |
| driver_substance_abuse | string | NONE DETECTED, IMPAIRED, ... | NONE DETECTED |
| driver_distracted_by | string | NOT DISTRACTED, CELL PHONE, ... | NOT DISTRACTED |
| vehicle_body_type | string | PASSENGER CAR, MOTORCYCLE, ... | PASSENGER CAR |
| vehicle_damage_extent | string | MINIMAL, FUNCTIONAL, ... | FUNCTIONAL |
| vehicle_movement | string | MOVING CONSTANT, STOPPED, ... | MOVING CONSTANT |
| speed_limit | integer | 0-120 (km/h) | 35 |
| driver_at_fault | string | "Yes", "No" | "Yes" |
| driverless_vehicle | string | "Yes", "No" | "No" |
| parked_vehicle | string | "Yes", "No" | "No" |
| vehicle_year | integer | 1980-2026 | 2020 |
| crash_hour | integer | 0-23 | 12 |
| crash_dayofweek | integer | 0 (pn) - 6 (nd) | 2 |
| crash_month | integer | 1-12 | 6 |
| crash_year | integer | 2015-2026 | 2026 |

---

## Statystyki opisowe (EDA)

### Dataset size
- **Wierszy:** 172,000
- **Kolumn (raw):** 43
- **Kolumn (po przetworzeniu):** ~40

### Wartości brakujące
- Weather: ~1%
- Light: <0.1%
- Surface Condition: ~2%
- Vehicle Year: ~0.5%

Strategia imputacji:
- **Kategorie:** tryb (moda)
- **Liczby:** mediana

### Zmienne kategoryczne
- Średnia liczba unikalnych wartości: 10-50
- Max: Weather (~8 wartości)
- Min: Light (~2 wartości)

### Zmienne numeryczne
- Speed Limit: mediana 35 km/h, range 0-120
- Vehicle Year: mediana 2015, range 1980-2025
- Latitude/Longitude: geograficznie Montgomery County, MD

---

## Odwołania do kodów kategorii

W surowych danych mogą być kody (np. "CODE_01"), które są zmapowane na opisowe wartości:

| Kod | Opis |
|-----|------|
| W1 | WEATHER = CLEAR |
| W2 | WEATHER = CLOUDY |
| W3 | WEATHER = RAIN |
| ... | ... |

(Pełne mapowanie można znaleźć w dokumentacji Montgomery County Open Data)

---

## Transformacje w pipeline'ie

1. **Usunięcie kolumn:** Report Number, Local Case Number, Location, itd.
2. **Imputacja braków:** tryb dla kategorii, mediana dla liczb
3. **Inżynieria cech:** godzina, dzień tygodnia, miesiąc, rok, wiek pojazdu
4. **Binarne cechy:** is_night, is_bad_weather, is_wet_surface
5. **Mapowanie klasy docelowej:** 5 wartości → 3 klasy (NO_INJURY, MINOR, SERIOUS)
6. **Enkodowanie:** wszystkie zmienne kategoryczne → liczby

---

## Zmienne w plikach pośrednich

### `data/02_intermediate/crash_cleaned.parquet`
Po czyszczeniu, przed inżynierią cech
- Brak wartości NaN
- Oryginalnie kodowane zmienne
- Oryginalna klasa docelowa (5 wartości)

### `data/03_primary/crash_features.parquet`
Po inżynierii cech i enkodowaniu
- Wszystkie wartości numeryczne (0, 1, 2, ..., -1)
- Cechy inżynieryjne (godzina, dzień, itp.)
- Zmapowana klasa docelowa (NO_INJURY, MINOR, SERIOUS)

### `data/04_feature/selected_features.parquet` (jeśli SelectKBest)
Podzbiór najważniejszych cech
- Wybrane tylko K cech
- Reszta analogiczna do _primary

---

## Znane problemy i ograniczenia

### Niezbalansowanie klas
```
NO_INJURY: 82%
MINOR:     17%
SERIOUS:   1%
```

**Wpływ na model:** Bias w kierunku NO_INJURY, gorsze predykcje SERIOUS

**Rozwiązania:**
- Stratified split (train/test)
- Class weights
- Metryka F1 macro zamiast accuracy

### Brakujące dane geograficzne
Latitude/Longitude są ustawione na 0.0 dla uproszczenia

### Kategorie w danych testowych
Jeśli w testach pojawią się nowe kategorie, są kodowane jako **-1** (UNKNOWN_CATEGORY_CODE)

---

## Schematy JSON

### CrashInput (API request)
```json
{
  "weather": "string",
  "light": "string",
  "collision_type": "string",
  "surface_condition": "string",
  "traffic_control": "string",
  "driver_substance_abuse": "string",
  "driver_distracted_by": "string",
  "vehicle_body_type": "string",
  "vehicle_damage_extent": "string",
  "vehicle_movement": "string",
  "speed_limit": "integer",
  "driver_at_fault": "string",
  "driverless_vehicle": "string",
  "parked_vehicle": "string",
  "vehicle_year": "integer",
  "crash_hour": "integer",
  "crash_dayofweek": "integer",
  "crash_month": "integer",
  "crash_year": "integer"
}
```

### PredictionOutput (API response)
```json
{
  "severity": "string (NO_INJURY | MINOR | SERIOUS | UNKNOWN)",
  "probabilities": {
    "NO_INJURY": "float",
    "MINOR": "float",
    "SERIOUS": "float"
  },
  "timestamp": "string (ISO 8601)"
}
```

---

## Podsumowanie

| Aspekt | Wartość |
|--------|---------|
| Zmienne w raw data | 43 |
| Zmienne po przetworzeniu | ~40 |
| Wierszy treningowych | ~137,600 (80%) |
| Wierszy testowych | ~34,400 (20%) |
| Klasy docelowe | 3 (NO_INJURY, MINOR, SERIOUS) |
| Dominująca klasa | NO_INJURY (82%) |
| Niezbalansowanie | Poważne (1% na SERIOUS) |
| Wartości brakujące | <3% (rozwiązane imputacją) |
| Typ problemu ML | Klasyfikacja wieloklasowa niezbalansowana |

