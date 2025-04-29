# 441project
Abstract
This study explores ML-Based Stock Analysis, integrating sentiment analysis and quantitative trading strategies to enhance investment decision-making. I collect financial news from Google News, perform sentiment analysis, and combine sentiment scores with stock price data. Using LSTM and GARCH, I predict stock price movements and optimize trading strategies.
Data sources：
news：https://news.google.com/
yahoo：https://finance.yahoo.com/


Overview

This project aims to analyze the relationship between news sentiment and stock returns. It collects news data and stock price data, performs sentiment analysis, aligns and cleans the data, and finally builds Transformer-based models to predict stock returns.

Data Collection
	•	News Data: Collected using GNews.
	•	Stock Data: Collected using yfinance.

Important:
Before running the scripts, make sure to update the file paths in the code to match your local directories where the data is stored.

Project Workflow
	1.	Run the scripts in the data folder
	•	Collect and prepare initial stock and news datasets.
	2.	Run emotion.py
	•	Perform sentiment analysis on the news articles and assign sentiment scores.
	3.	Run data_alignment.py
	•	Merge sentiment scores with stock data, aligning by date and stock ticker.
	4.	Run save_fill_random_sentiment.py
	•	Fill missing sentiment values with randomly sampled scores to ensure completeness.
	5.	Run modeling and evaluation scripts
	•	Final_Stock_Sentiment_Analysis_and_Transformer_Model.py
	•	Conduct Central Limit Theorem (CLT) tests, hypothesis testing, and train a Transformer model to predict Apple’s stock returns.
	•	multi_stock_transformer_evaluation.py
	•	Train and evaluate Transformer models across ten different stocks, comparing prediction performance.

Requirements
	•	Python 3.8+
	•	Packages:
pandas, numpy, matplotlib, scipy, scikit-learn, tensorflow, yfinance, gnews

You can install the necessary packages with:

pip install pandas numpy matplotlib scipy scikit-learn tensorflow yfinance gnews

Notes
	•	Always check and adjust file paths for your local environment.
	•	Sentiment analysis is performed at the news article level.
	•	Transformer models are trained separately for single stock (Apple) and multiple stocks.
