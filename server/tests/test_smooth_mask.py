from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def test_smooth_mask():
    with open("tests/images/test_mask.png", "rb") as image_file:
        response = client.post(
            "/smooth_mask/",
            data={
                "project_id": "test_404176_2_0_1000"
            },
            files={"image": ("tests/images/smooth_mask.png", image_file, "image/png")}
        )

    assert response.status_code == 200
    response_data = response.json()

    assert response_data["message"] == "Image uploaded successfully"
    assert response_data["url"] == "https://daikan-data-devo2.s3.ap-northeast-1.amazonaws.com/test_404176_2_0_1000/smoothed.png"
    assert response_data["presigned_url"].startswith("https://daikan-data-devo2.s3.amazonaws.com/")
    assert response_data["image_base64"].startswith("iVBORw0KGgoAAAANSUhEUgAAA+gAAAEuC")
