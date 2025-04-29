import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_csv("Final_Stock_Sentiment_Returns.csv")

# 查看有多少缺失
num_missing = df["Sentiment_Score"].isna().sum()
print(f"Sentiment_Score 缺失数量: {num_missing}")

# 如果有缺失，就随机填充
if num_missing > 0:
    random_values = np.random.uniform(0, 1, size=num_missing)
    df.loc[df["Sentiment_Score"].isna(), "Sentiment_Score"] = random_values
    print(f"✅ 已用随机数填充 {num_missing} 个缺失值。")
else:
    print("没有缺失，无需填充。")

# 保存处理后的数据
df.to_csv("Final_Stock_Sentiment_Returns_Filled_Random.csv", index=False)
print("✅ 完成保存，文件名：Final_Stock_Sentiment_Returns_Filled_Random.csv")