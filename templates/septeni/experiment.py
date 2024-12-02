import argparse
import json
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Process and train a linear regression model.")
parser.add_argument("--file_path", type=str, default="レポートAI用データ - AI用.csv", help="Path to the input CSV file.")
parser.add_argument("--out_dir", type=str, default="run_0", help="Directory to save the output results (default: run_0)")

args = parser.parse_args()

# 引数からパラメータを取得
file_path = args.file_path
out_dir = args.out_dir

# データ読み込み
# file_path = '/mnt/data/レポートAI用データ - AI用.csv'
data = pd.read_csv(file_path)

# 必要なカラムの選択と前処理
columns_to_use = ['在籍日数', '4特性', 'レビュー数', '良かった点の数', 
                  '情報収集力', '継続力', '論理的思考', 'リーダーシップ', 
                  '創造性', '冷静な対応', '顧客志向', '進行管理', '本質を理解する力', '対象スコア']
data = data[columns_to_use]

# データの前処理: 数値変換や欠損値処理
data['4特性'] = data['4特性'].astype('category').cat.codes  # カテゴリを数値に変換
data = data.replace({'%': ''}, regex=True).astype(float)  # パーセントを数値化
data = data.dropna()  # 欠損値を削除

# 特徴量とターゲット変数に分割
X = data.drop(columns=['対象スコア'])
y = data['対象スコア']

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの訓練
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 結果の評価
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 結果をJSONファイルに保存
final_results = {
    "Mean Absolute Error (MAE)": {"means": mae},
    "Mean Squared Error (MSE)": {"means": mse},
    "R-squared (R2)": {"means": r2}
}

os.makedirs(out_dir, exist_ok=True)

# 結果をJSONファイルとして保存
output_path = os.path.join(out_dir, "final_info.json")
with open(output_path, "w") as f:
    json.dump(final_results, f, indent=4)