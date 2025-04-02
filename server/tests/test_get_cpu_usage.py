from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_cpu_usage():
    response = client.get("/get_cpu_usage/")
    assert response.status_code == 200
    json_data = response.json()

    assert "cpu_usage" in json_data
    assert "announcement" in json_data

    cpu_usage = json_data["cpu_usage"]
    announcement = json_data["announcement"]

    assert isinstance(cpu_usage, float)
    assert 0.0 <= cpu_usage <= 100.0

    if cpu_usage <= 30:
        assert announcement == "空いている"
    elif 30 < cpu_usage <= 70:
        assert announcement == "普通"
    else:
        assert announcement == "混んでいる"
