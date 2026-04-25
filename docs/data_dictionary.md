# Dane - crash_data.csv

Zbiór danych pochodzi z Montgomery County, Maryland (Crash Reporting Incidents, 2015-2024).

## Wymiary
- 172 105 wierszy
- 43 kolumny

## Target

`Injury Severity` mapowany na 3 klasy `Severity_Group`:
- NO_INJURY (~82%)
- MINOR (~17%)
- SERIOUS (~1%)

## Wybrane cechy używane w modelu

- Weather - warunki pogodowe
- Light - oświetlenie
- Surface Condition - stan nawierzchni
- Collision Type - typ kolizji
- Speed Limit - limit prędkości
- Traffic Control - kontrola ruchu
- Vehicle Body Type, Vehicle Make/Model, Vehicle Year
- Vehicle Damage Extent - zakres uszkodzeń
- Driver Substance Abuse, Driver Distracted By
- Crash Date/Time - z tego wyciągamy godzinę, dzień tygodnia, miesiąc

## Cechy odrzucone

Identyfikatory (Report Number, Person ID itd.) i kolumny z >90% braków
(Off-Road Description, Related Non-Motorist).

## Cechy pochodne (feature engineering)

- crash_hour, crash_dayofweek, crash_month, crash_year - z daty
- is_night - czy DARK w Light
- is_bad_weather - czy pogoda inna niż CLEAR/CLOUDY
- is_wet_surface - czy nawierzchnia inna niż DRY
- vehicle_age - wiek pojazdu
