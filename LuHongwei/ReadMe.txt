Project Workflow
	1.	Run the scripts in the data folder
	•	These scripts are used to collect and prepare the initial stock and news data.
	2.	Perform sentiment analysis using emotion.py
	•	This script assigns sentiment scores to the news articles based on their content.
	3.	Align sentiment scores with stock data using data_alignment.py
	•	This step merges the sentiment scores with the corresponding stock data, ensuring that each score matches the correct stock and date.
	4.	Handle missing sentiment values with save_fill_random_sentiment.py
	•	This script fills missing sentiment values with randomly sampled scores to maintain dataset integrity.
	5.	Final analysis and modeling
	•	Final_Stock_Sentiment_Analysis_and_Transformer_Model.py:
Performs CLT (Central Limit Theorem) tests, hypothesis testing, and trains a Transformer model to predict Apple’s stock returns.
	•	multi_stock_transformer_evaluation.py:
Trains and compares Transformer models across ten different stocks for performance evaluation.
When running the code, please make sure to update the file paths to match your local data directories.



pandas
numpy
matplotlib
scipy
scikit-learn
tensorflow
transformers
arch
gnews
yfinance