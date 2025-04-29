from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd

# 载入数据
news_df = pd.read_csv("GoogleNews_Stock_News.csv")

# 加载 FinBERT 模型
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 给每条新闻打标签（用Title或者Description）
def get_sentiment(text):
    try:
        result = nlp(text)[0]
        return result["label"], result["score"]
    except:
        return None, None  # 防止出错

# 对新闻标题进行情绪打分
news_df[["Sentiment_Label", "Sentiment_Score"]] = news_df["Title"].apply(lambda x: pd.Series(get_sentiment(x)))

# 保存带情绪分析的新文件
news_df.to_csv("GoogleNews_Stock_News_with_Sentiment.csv", index=False)
print("新闻情绪分析完成并保存！")
#打分后发现有很多缺失值，所以选择了用随机数的方法，就有了save_fill_random_sentiment