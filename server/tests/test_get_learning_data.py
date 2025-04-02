from fastapi.testclient import TestClient
from app.main import app
import json
import pandas as pd
from unittest.mock import patch
import pytest

client = TestClient(app)

# モックデータ
mock_data_hyomen = pd.DataFrame({
    'pj': [1, 2, 3, 1]
})
mock_data_uramen = pd.DataFrame({
    'pj': [1, 2, 2, 3]
})
mock_data_sokumen = pd.DataFrame({
    'pj': [4, 5, 6]
})
mock_data_typea = pd.DataFrame({
    'pj': [7, 8, 7, 8]
})
mock_data_urasokumen = pd.DataFrame({
    'pj': [7, 8, 7, 8]
})
mock_data_neon = pd.DataFrame({
    'pj': [7, 8, 7, 8]
})

# get_unique_pj_countをモックするためのパッチ
@pytest.fixture
def mock_read_csv():
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = [
            mock_data_hyomen,  # hyomen
            mock_data_uramen,  # uramen
            mock_data_sokumen,  # sokumen
            mock_data_typea, # typea
            mock_data_urasokumen,  # urasokumen
            mock_data_neon
        ]
        yield mock_read_csv


# テストケース
def test_get_learning_data(mock_read_csv):
    # APIにPOSTリクエストを送信
    response = client.post("/get_learning_data/")

    # レスポンスのステータスコードを確認
    assert response.status_code == 200

    # レスポンスデータの内容を確認
    response_data = response.json()

    # 各フィールドの "number" が正しい値であるか確認
    assert response_data["hyomen"]["number"] == 4
    assert response_data["uramen"]["number"] == 4
    assert response_data["sokumen"]["number"] == 3
    assert response_data["typea"]["number"] == 4

    assert isinstance(response_data["hyomen"]["number_of_under_5"], int)
    assert isinstance(response_data["uramen"]["number_of_under_5"], int)
    assert isinstance(response_data["sokumen"]["number_of_under_5"], int)
    assert isinstance(response_data["typea"]["number_of_under_5"], int)

    assert isinstance(response_data["hyomen"]["number_of_under_5_to_10"], int)
    assert isinstance(response_data["uramen"]["number_of_under_5_to_10"], int)
    assert isinstance(response_data["sokumen"]["number_of_under_5_to_10"], int)
    assert isinstance(response_data["typea"]["number_of_under_5_to_10"], int)

    assert isinstance(response_data["hyomen"]["number_of_over_10"], int)
    assert isinstance(response_data["uramen"]["number_of_over_10"], int)
    assert isinstance(response_data["sokumen"]["number_of_over_10"], int)
    assert isinstance(response_data["typea"]["number_of_over_10"], int)

