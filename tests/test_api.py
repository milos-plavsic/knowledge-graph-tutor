from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").status_code == 200


def test_explain() -> None:
    r = client.post("/v1/explain", json={"topic": "Gradients"})
    assert r.status_code == 200
    assert r.json()["topic"] == "Gradients"
    assert "reasoning path" in r.json()["explanation"]
