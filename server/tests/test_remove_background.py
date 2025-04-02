from fastapi.testclient import TestClient
from app.main import app
import hashlib
import json

client = TestClient(app)

def test_remove_background():
    data = {
        "project_id": "test_404176_2_0_1000",
        "algorithm": 1
    }
    data_to_send = {
        "request_body": json.dumps(data)
    }

    with open("tests/images/test_404176_2_0_1000.png", "rb") as image_file:
        response = client.post(
            "/remove_background/",
            data=data_to_send,
            files={"image": ("tests/images/test_404176_2_0_1000.png", image_file, "image/png")}
        )
    assert response.status_code == 200
    response_data = response.json()

    assert response_data["message"] == "Image uploaded successfully"
    assert response_data["url"] == "https://daikan-data-devo2.s3.ap-northeast-1.amazonaws.com/test_404176_2_0_1000/segmented.png"
    assert response_data["presigned_url"].startswith("https://daikan-data-devo2.s3.amazonaws.com/")
    base64_img = response_data["image_base64"].encode('utf-8')
    hash_value = hashlib.sha256(base64_img).hexdigest()
    assert hash_value == "31b91d25a2e3c2eb1c5d1db1d1508c821179090161a0b132f6c6a026fbd635a8"
