import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. 讀取與檢查資料 ===
# 請確認活頁簿1.pdf 中的資料已轉成適合分析的表格格式（例如 CSV 或 Excel）
# 這裡假設已轉換為 CSV 檔案，名稱為 "活頁簿1.csv"
data = pd.read_csv("活頁簿1.csv")

# 檢查資料結構與前 5 筆記錄
print("資料欄位：", data.columns.tolist())
print("前五筆資料：")
print(data.head())

# === 2. 基本資料描述性統計 ===
basic_vars = ['Grade', 'MathInterest', 'MathBase']
print("\n基本資料描述性統計：")
print(data[basic_vars].describe())

print("\n各分類變項頻率：")
for var in basic_vars:
    print(f"{var} 頻率分布：")
    print(data[var].value_counts())
    print()

# === 3. Likert 量表題描述性統計 ===
likert_cols = ['Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14']
print("\nLikert 量表題描述性統計：")
print(data[likert_cols].describe())

# === 4. Cronbach's alpha 計算 ===
def cronbach_alpha(df):
    """
    計算 Cronbach's alpha 以評估量表內部一致性
    """
    df = df.dropna()  # 移除缺失值
    k = df.shape[1]
    item_vars = df.var(ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    alpha = k / (k - 1) * (1 - item_vars.sum() / total_var)
    return alpha

alpha_value = cronbach_alpha(data[likert_cols])
print("\nCronbach's alpha 值：", round(alpha_value, 2))

# === 5. 群組比較 ===
# 由於 Grade 全部為 3，此處示範依 MathInterest 分組
# 可將 MathInterest 分成高興趣 (≥4) 與低興趣 (<4) 兩組
data['Interest_Group'] = data['MathInterest'].apply(lambda x: 'High' if x >= 4 else 'Low')
grouped_interest = data.groupby('Interest_Group')[likert_cols].mean()
print("\n依 MathInterest 分組的 Likert 題目平均分數：")
print(grouped_interest)

# === 6. 視覺化示範 ===
# (a) 各 Likert 題目的直方圖
for col in likert_cols:
    plt.figure()
    plt.hist(data[col].dropna(), bins=np.arange(1,7)-0.5, edgecolor='black')
    plt.title(f"{col} 分數分布")
    plt.xlabel("分數")
    plt.ylabel("人數")
    plt.xticks(range(1,6))
    plt.show()

# (b) Likert 題目相關矩陣
corr_matrix = data[likert_cols].corr()
plt.figure(figsize=(8,6))
plt.imshow(corr_matrix, cmap='viridis', interpolation='none')
plt.colorbar()
plt.xticks(range(len(likert_cols)), likert_cols, rotation=90)
plt.yticks(range(len(likert_cols)), likert_cols)
plt.title("Likert 題目相關矩陣")
plt.show()

