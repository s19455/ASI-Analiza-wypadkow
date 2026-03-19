# Slownik danych - crash_data.csv

## Zrodlo
Montgomery County, Maryland - Crash Reporting Incidents (2015-2024)

## Kolumny (43)

| # | Kolumna | Typ | Opis | Uzyta w modelu |
|---|---------|-----|------|----------------|
| 1 | Report Number | string | Numer raportu policyjnego | Nie (identyfikator) |
| 2 | Local Case Number | string | Lokalny numer sprawy | Nie (identyfikator) |
| 3 | Agency Name | string | Nazwa agencji raportującej | Nie (stała) |
| 4 | ACRS Report Type | string | Typ raportu (Property Damage/Injury/Fatal) | Nie (koreluje z targetem) |
| 5 | Crash Date/Time | datetime | Data i czas wypadku | Tak (extracted: hour, dayofweek, month, year) |
| 6 | Route Type | string | Typ drogi (Maryland, US, County) | Tak |
| 7 | Road Name | string | Nazwa drogi | Tak (label encoded) |
| 8 | Cross-Street Type | string | Typ skrzyzowania | Tak |
| 9 | Cross-Street Name | string | Nazwa ulicy krzyżowej | Tak (label encoded) |
| 10 | Off-Road Description | string | Opis zjazdu z drogi | Nie (91% brakujących) |
| 11 | Municipality | string | Gmina | Tak |
| 12 | Related Non-Motorist | string | Powiązany pieszy/rowerzysta | Nie (97% brakujących) |
| 13 | Collision Type | string | Typ kolizji | Tak |
| 14 | Weather | string | Warunki pogodowe | Tak + derived: is_bad_weather |
| 15 | Surface Condition | string | Stan nawierzchni | Tak + derived: is_wet_surface |
| 16 | Light | string | Oświetlenie | Tak + derived: is_night |
| 17 | Traffic Control | string | Kontrola ruchu (sygnalizacja, znaki) | Tak |
| 18 | Driver Substance Abuse | string | Substancje odurzające kierowcy | Tak |
| 19 | Non-Motorist Substance Abuse | string | Substancje pieszego | Nie (97% brakujących) |
| 20 | Person ID | string | ID osoby | Nie (identyfikator) |
| 21 | Driver At Fault | string | Czy kierowca zawinił (Yes/No) | Tak |
| 22 | **Injury Severity** | string | **TARGET** - stopień obrażeń | Tak (mapowany na 3 klasy) |
| 23 | Circumstance | string | Okoliczności wypadku | Tak |
| 24 | Driver Distracted By | string | Przyczyna rozproszenia kierowcy | Tak |
| 25 | Drivers License State | string | Stan prawa jazdy | Tak |
| 26 | Vehicle ID | string | ID pojazdu | Nie (identyfikator) |
| 27 | Vehicle Damage Extent | string | Zakres uszkodzeń pojazdu | Tak |
| 28 | Vehicle First Impact Location | string | Miejsce pierwszego uderzenia | Tak |
| 29 | Vehicle Second Impact Location | string | Miejsce drugiego uderzenia | Tak |
| 30 | Vehicle Body Type | string | Typ nadwozia pojazdu | Tak |
| 31 | Vehicle Movement | string | Ruch pojazdu | Tak |
| 32 | Vehicle Continuing Dir | string | Kierunek kontynuacji jazdy | Tak |
| 33 | Vehicle Going Dir | string | Kierunek jazdy | Tak |
| 34 | Speed Limit | int | Limit prędkości (mph) | Tak |
| 35 | Driverless Vehicle | string | Pojazd bez kierowcy | Tak |
| 36 | Parked Vehicle | string | Pojazd zaparkowany | Tak |
| 37 | Vehicle Year | int | Rok produkcji pojazdu | Tak + derived: vehicle_age |
| 38 | Vehicle Make | string | Marka pojazdu | Tak (label encoded) |
| 39 | Vehicle Model | string | Model pojazdu | Tak (label encoded) |
| 40 | Equipment Problems | string | Problemy z wyposażeniem | Tak |
| 41 | Latitude | float | Szerokość geograficzna | Tak |
| 42 | Longitude | float | Długość geograficzna | Tak |
| 43 | Location | string | Lokalizacja (tekst) | Nie (redundant z lat/long) |

## Zmienna docelowa

**Injury Severity** -> mapowana na **Severity_Group** (3 klasy):

| Severity_Group | Oryginalne wartości | Udział |
|----------------|--------------------| -------|
| NO_INJURY | NO APPARENT INJURY | ~82% |
| MINOR | POSSIBLE INJURY, SUSPECTED MINOR INJURY | ~17% |
| SERIOUS | SUSPECTED SERIOUS INJURY, FATAL INJURY | ~1% |

## Cechy pochodne (feature engineering)

| Cecha | Źródło | Opis |
|-------|--------|------|
| crash_hour | Crash Date/Time | Godzina wypadku (0-23) |
| crash_dayofweek | Crash Date/Time | Dzień tygodnia (0=Pon, 6=Nd) |
| crash_month | Crash Date/Time | Miesiąc (1-12) |
| crash_year | Crash Date/Time | Rok |
| is_night | Light | 1 jeśli "DARK" w warunkach oświetlenia |
| is_bad_weather | Weather | 1 jeśli pogoda inna niż CLEAR/CLOUDY |
| is_wet_surface | Surface Condition | 1 jeśli nawierzchnia inna niż DRY |
| vehicle_age | Vehicle Year | 2026 - Vehicle Year (clip 0-50) |
