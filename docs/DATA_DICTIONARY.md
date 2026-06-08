# Słownik danych

Opis zmiennych w zbiorze o wypadkach drogowych z Montgomery County, Maryland (2015-2024).

## Dane wejściowe (raw)

- **Źródło:** Montgomery County Open Data (Maryland)
- **Format:** CSV
- **Ścieżka:** `data/01_raw/crash_data.csv`
- **Rozmiar:** ~172 000 wierszy × 43 kolumny

## Wybrane kolumny surowe

### Informacje o wypadku
| Kolumna | Typ | Opis |
|---------|-----|------|
| Report Number | string | Identyfikator raportu |
| Crash Date/Time | datetime | Data i czas wypadku |
| Agency Name | string | Agencja raportująca |
| Location | string | Opis lokalizacji |

### Warunki
| Kolumna | Typ | Przykłady |
|---------|-----|-----------|
| Weather | string | CLEAR, CLOUDY, RAINING, FOGGY, SNOW |
| Light | string | DAYLIGHT, DARK LIGHTS ON, DARK NO LIGHTS, DUSK |
| Surface Condition | string | DRY, WET, SNOW, ICE |
| Traffic Control | string | NO CONTROLS, TRAFFIC SIGNAL, STOP SIGN |

### Kierowca
| Kolumna | Typ | Przykłady |
|---------|-----|-----------|
| Driver Substance Abuse | string | NONE DETECTED, ALCOHOL PRESENT, ... |
| Driver At Fault | string | Yes, No |
| Driver Distracted By | string | NOT DISTRACTED, TEXTING FROM A CELLULAR PHONE, ... |
| Drivers License State | string | MD, VA, ... |

### Pojazd
| Kolumna | Typ | Przykłady |
|---------|-----|-----------|
| Vehicle Year | integer | 2015, 2020 |
| Vehicle Make / Model | string | TOYOTA / CIVIC |
| Vehicle Body Type | string | PASSENGER CAR, (SPORT) UTILITY VEHICLE, PICKUP TRUCK |
| Vehicle Movement | string | MOVING CONSTANT SPEED, STOPPED IN TRAFFIC LANE, MAKING LEFT TURN |
| Vehicle Damage Extent | string | FUNCTIONAL, DISABLING, DESTROYED |

### Target
| Kolumna | Typ | Wartości |
|---------|-----|----------|
| **Injury Severity** (target) | string | NO APPARENT INJURY, POSSIBLE INJURY, SUSPECTED MINOR INJURY, SUSPECTED SERIOUS INJURY, FATAL INJURY |

## Dane po przetworzeniu

**Ścieżka:** `data/03_primary/crash_features.parquet` (pipeline `data_preparation`)

### Cechy inżynieryjne
| Kolumna | Typ | Zakres |
|---------|-----|--------|
| crash_hour | integer | 0-23 |
| crash_dayofweek | integer | 0 (pn) - 6 (nd) |
| crash_month | integer | 1-12 |
| crash_year | integer | 2015-2024 |
| is_night | integer | 0/1 (Light zawiera DARK) |
| is_bad_weather | integer | 0/1 (nie CLEAR/CLOUDY) |
| is_wet_surface | integer | 0/1 (nie DRY) |
| vehicle_age | integer | 0-50 |

### Zmienne kategoryczne (enkodowane)

Każda kolumna tekstowa jest mapowana na liczby całkowite `0, 1, 2, ...`. Wartości nieznane (niewidziane podczas treningu) dostają kod `-1`. Mapowania są zapisane w `data/06_models/encoders.pkl`.

### Zmienna docelowa

`Injury Severity` (5 wartości) jest mapowana na `Severity_Group` (3 klasy):

```yaml
NO_INJURY: [NO APPARENT INJURY]
MINOR:     [POSSIBLE INJURY, SUSPECTED MINOR INJURY]
SERIOUS:   [SUSPECTED SERIOUS INJURY, FATAL INJURY]
```

Rozkład klas: NO_INJURY ~82%, MINOR ~17%, SERIOUS ~1%.

## Transformacje w pipeline'ie

1. Usunięcie kolumn (Report Number, Local Case Number, Location, ...)
2. Imputacja braków (moda dla tekstu, mediana dla liczb)
3. Inżynieria cech (czas, wiek pojazdu, cechy binarne)
4. Mapowanie targetu na 3 klasy
5. Enkodowanie zmiennych kategorycznych

## Uwagi

- **Niezbalansowanie:** klasa SERIOUS to ~1% danych — używamy stratified split, `class_weight="balanced"` i F1 makro.
- **Dane geograficzne:** Latitude/Longitude w inferencji API ustawiane są na 0.0 (uproszczenie).
- **Nieznane kategorie:** kodowane jako `-1`.

## Podział train/test

| | Wiersze |
|---|---|
| Treningowe (80%) | ~137 600 |
| Testowe (20%) | ~34 400 |
