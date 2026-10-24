import pytest
from fastapi.testclient import TestClient

from powercast.api.main import app
from powercast.config import REPORTS

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_rejects_unknown_zone():
    assert client.get("/forecast", params={"zone": "XX"}).status_code == 422


@pytest.mark.skipif(not (REPORTS / "predictions.parquet").exists(), reason="backtest not built")
def test_backtest_day_has_24_ordered_quantiles():
    r = client.get("/forecast", params={"zone": "NL", "date": "2025-06-15"})
    assert r.status_code == 200
    hours = r.json()["hours"]
    assert len(hours) == 24
    assert all(h["q10"] <= h["q50"] <= h["q90"] for h in hours)
