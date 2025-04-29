import yfinance as yf
import pandas as pd

# 定义要收集的10支股票
stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "JPM", "GS", "XOM", "PFE", "KO"]

# 下载股票数据（比如最近两年）
stock_data = yf.download(stocks, start="2016-01-01", end="2025-04-20")

# 保存为CSV
stock_data.to_csv("YahooFinance_Stock_Data.csv")
print("✅ 股票市场数据已保存！")