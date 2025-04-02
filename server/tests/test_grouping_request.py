from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def test_grouping_request():
    # テストデータを作成
    request_data = {
        "image_dictionary": {
            "1": [
                    [1, 1, 1],
                    [0, 1, 0],
                    [1, 0, 1],
            ],
            "2": [
                    [1, 1, 1],
                    [1, 0, 1],
                    [0, 1, 0]
                ]
        },
        "scale_ratio": 1.0
    }
    
    # APIにPOSTリクエストを送信
    response = client.post("/grouping_request/", json=request_data)
    
    # ステータスコードが200であることを確認
    assert response.status_code == 200
    
    # レスポンスデータを取得
    response_data = response.json()
    
    # 結果を確認
    assert "height" in response_data
    assert "width" in response_data
    
    # 期待される結果
    expected_max_height = 2
    expected_max_width = 3
    
    assert response_data["height"] == expected_max_height
    assert response_data["width"] == expected_max_width