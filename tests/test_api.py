import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_regions_endpoint():
    r = client.get("/meta/regions")
    assert r.status_code == 200
    assert "Global" in r.json()
