from gnews import GNews
import pandas as pd

# 初始化 GNews对象
google_news = GNews(language='en', country='US', max_results=1000)  # 每支股票最多100条新闻

# 定义目标股票列表
stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "JPM", "GS", "XOM", "PFE", "KO"]

# 创建空DataFrame
news_df = pd.DataFrame(columns=["Stock", "Title", "Description", "Published Date", "Link"])

# 抓取每支股票的新闻
for stock in stocks:
    news = google_news.get_news(f"{stock} stock")
    for item in news:
        news_df = news_df.append({
            "Stock": stock,
            "Title": item["title"],
            "Description": item["description"],
            "Published Date": item["published date"],
            "Link": item["url"]
        }, ignore_index=True)
    print(f"✅ {stock} 新闻数据收集完成")

# 保存所有新闻为CSV文件
news_df.to_csv("GoogleNews_Stock_News.csv", index=False)
print("✅ 所有股票的新闻数据已保存为 GoogleNews_Stock_News.csv")