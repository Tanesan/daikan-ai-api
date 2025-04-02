import sys

import requests
import csv
from collections import defaultdict

# APIのエンドポイント
api_url = "http://localhost:8006/predict_led"

tsv_name = "nishikawa11-16hyomen"
# 読み込むTSVファイルのパス
input_tsv = f"{tsv_name}.tsv"
# 書き込むTSVファイルのパス
output_tsv = f"{tsv_name}_add_prediction.tsv"


# TSVファイルの読み込みとpj毎のデータをグループ化
pj_groups = defaultdict(list)
with open(input_tsv, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile, delimiter='\t')
    fieldnames = reader.fieldnames + ["processed_path", 'skeleton_length', 'intersection3', 'intersection4', 'intersection5', 'intersection6', 'endpoint', 'Area', 'Peri', 'distance', 'nomal', "recommendation"]  # 追加するカラム名
    rows = [row for row in reader]
    for row in rows:
        pj_groups[row['pj']].append(row)

# pj毎にAPIリクエストを送信
for pj, group_rows in pj_groups.items():
        # リクエスト用のデータ作成
        perimages = {}
        for idx, row in enumerate(group_rows):
            prop_name = f"{idx+1}"

            # TSVからscale_ratioとdistanceを取得
            scale_ratio = float(row['scale_ratio'])
            distance = list(map(float, row['distance'].strip('[]').split(',')))
            perimages[prop_name] = {
                "image": [[0]],  # 空のimage配列
                "height":0,
                "width": 0,
                "area": row["Area"],
                "area_for_frontend": 1,
                "perimeter_for_frontend": 1,
                "peri": row["Peri"],
                "x": 0,
                "y": 0,
                "distance": distance,
                "skeleton_length": row['skeleton_length'],
                "intersection_count3": row["intersection3"],
                "intersection_count4": row["intersection4"],
                "intersection_count5": row["intersection5"],
                "intersection_count6": row["intersection6"],
                "endpoints_count": row["endpoint"],
                "luminous_model": 1
            }

        # APIリクエストデータ
        data = {
            "height": row["processed_path"].split("_")[2],
            "width": 0,
            "scale_ratio": 1,  # 同じpjのレコードで同一のscale_ratio
            "scale_ratio_for_frontend_scale": 1,
            "perimages": perimages
        }

        # APIを呼び出して結果を取得
        response = requests.post(api_url, json=data)
        result_json = response.json()
        print(result_json)
        for idx, row in enumerate(group_rows):
            row["nomal"] = result_json[f"{idx + 1}"]["nomal"]


# 新しいTSVファイルに書き込み
with open(output_tsv, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)