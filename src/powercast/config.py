"""Project-wide settings: zones, weather points, paths and the train/val/test split."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
REPORTS = ROOT / "reports"

ZONES = ["NL", "DE-LU", "FR"]

# Delivery days are defined in market time (CET/CEST). The day-ahead auction for
# day D closes at 12:00 market time on D-1.
MARKET_TZ = "Europe/Brussels"

# Weather points per zone: (name, lat, lon). Mixed demand centres and wind/solar regions.
WEATHER_POINTS = {
    "NL": [("amsterdam", 52.37, 4.89), ("rotterdam", 51.92, 4.48), ("groningen", 53.22, 6.57)],
    "DE-LU": [
        ("hamburg", 53.55, 9.99), ("berlin", 52.52, 13.40), ("frankfurt", 50.11, 8.68),
        ("munich", 48.14, 11.58), ("husum", 54.48, 9.05),
    ],
    "FR": [
        ("paris", 48.86, 2.35), ("lyon", 45.76, 4.84), ("marseille", 43.30, 5.37),
        ("lille", 50.63, 3.06), ("nantes", 47.22, -1.55),
    ],
}
WEATHER_VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]

# Holiday calendars (python-holidays country codes) per zone.
HOLIDAY_COUNTRY = {"NL": "NL", "DE-LU": "DE", "FR": "FR"}

HISTORY_START = "2019-01-01"
TRAIN_END = "2023-12-31"
VAL_START, VAL_END = "2024-01-01", "2024-12-31"
TEST_START = "2025-01-01"

# Battery used in the trading backtest.
BATTERY = dict(power_mw=1.0, energy_mwh=2.0, round_trip_eff=0.90, max_cycles_per_day=1.0)
