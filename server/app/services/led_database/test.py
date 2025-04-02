import pandas as pd


file_name ="predict_database.tsv"
# ファイルの読み込み
df = pd.read_csv(file_name, sep='\t')
df1 = pd.read_csv('hyomen_adjust_1.tsv', sep='\t')
# df1 = df1.drop(["predict_led"], axis=1)
print(df1)
df_merged = pd.merge(df, df1, on=["pj", "index"], how="outer", suffixes=("", "_B"))

# df_Bに値がある場合、Aを上書きする（例えば、"index"以外のカラムが複数あると仮定）
for col in df1.columns:
    if col not in ["pj", "index"]:
        df_merged[col] = df_merged[col + "_B"].combine_first(df_merged[col])

# 不要なBのカラムを削除
df_merged = df_merged[df.columns]

# 結果を確認
df_A_updated = df_merged
print(len(df_A_updated))
# df = df.drop(columns=["recommend"], axis=1)

# 結果をCSVに保存
df_A_updated.to_csv(file_name, index=False, sep="\t")

