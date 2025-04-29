import pandas as pd

# 读取情绪数据和股票数据
sentiment_df = pd.read_csv("GoogleNews_Stock_News_with_Sentiment.csv")
stock_df = pd.read_csv("YahooFinance_Stock_Data.csv", header=[0,1], index_col=0)

# 转换日期格式
sentiment_df["Published Date"] = pd.to_datetime(sentiment_df["Published Date"]).dt.date

# 处理股票价格数据
# 提取收盘价（Close）部分
close_price = stock_df["Close"]

# 将索引转为日期格式
close_price.index = pd.to_datetime(close_price.index).date

# 计算每日收益率（Return）
returns = close_price.pct_change()

# 保存收益率数据
returns.to_csv("Stock_Returns.csv")
print("股票收益率计算完成！")

# 统计每天的平均情绪得分
daily_sentiment = sentiment_df.groupby(["Stock", "Published Date"])["Sentiment_Score"].mean().reset_index()

# 合并每日情绪得分和股票收益率
final_df = pd.DataFrame()

for stock in close_price.columns:
    # 挑选对应股票
    stock_sentiment = daily_sentiment[daily_sentiment["Stock"] == stock]
    stock_sentiment = stock_sentiment.rename(columns={"Published Date": "Date"})
    
    # 挑选对应股票收益率
    stock_return = returns[stock].reset_index()
    stock_return.columns = ["Date", "Return"]
    
    # 合并情绪和收益率
    merged = pd.merge(stock_return, stock_sentiment, on="Date", how="left")
    merged["Stock"] = stock
    
    final_df = pd.concat([final_df, merged], axis=0)

# 保存最终数据
final_df.to_csv("Final_Stock_Sentiment_Returns.csv", index=False)
print("股票收益率和情绪数据对齐完成！文件保存为 Final_Stock_Sentiment_Returns.csv")