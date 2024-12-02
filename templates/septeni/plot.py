import json

import matplotlib.pyplot as plt
import pandas as pd

# final_info.json ファイルを読み込む
input_path = 'run_0/final_info.json'

with open(input_path, 'r') as f:
    final_results = json.load(f)

# データの準備
metrics = list(final_results.keys())
values = [final_results[metric]["means"] for metric in metrics]

# プロットを作成
plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color='skyblue')
plt.title('Performance Metrics from final_info.json', fontsize=16)
plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("final_info.png")
# plt.show()


data_path = 'レポートAI用データ - AI用.csv'
data = pd.read_csv(data_path)

# 必要なカラムを選択して欠損値を除外
columns_to_use = ['在籍日数', '4特性', 'レビュー数', '良かった点の数', '対象スコア']
data = data[columns_to_use]
data['4特性'] = data['4特性'].astype('category').cat.codes  # カテゴリを数値化
data = data.dropna()

plt.figure(figsize=(8, 6))

# 散布図 (在籍日数 vs 対象スコア)
plt.scatter(data['在籍日数'], data['対象スコア'], alpha=0.7, c=data['4特性'], cmap='viridis')
plt.colorbar(label='4特性')
plt.title('Scatter Plot: 在籍日数 vs 対象スコア', fontsize=16)
plt.xlabel('在籍日数', fontsize=14)
plt.ylabel('対象スコア', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# プロットの保存
plt.tight_layout()
plt.savefig("scatter.png")
# plt.show()