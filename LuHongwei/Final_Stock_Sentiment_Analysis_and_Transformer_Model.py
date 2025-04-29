# ==========================================
# Final_Stock_Sentiment_Returns Full Analysis Pipeline
# Including: Data Cleaning, CLT Verification, Hypothesis Testing, Feature Engineering, Transformer Regression
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, mannwhitneyu, ks_2samp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# Step 1: Data Cleaning
# ==========================================
print("\n=== Step 1: Data Cleaning ===")

# Read the original dataset
df = pd.read_csv("Final_Stock_Sentiment_Returns_Filled_Random.csv")

# Drop rows with missing Return or Sentiment_Score
df = df.dropna(subset=["Return", "Sentiment_Score"])

# Filter out outliers
df = df[(df["Return"] > -0.5) & (df["Return"] < 0.5)]
df = df[(df["Sentiment_Score"] >= 0) & (df["Sentiment_Score"] <= 1)]

# Format Date column
df["Date"] = pd.to_datetime(df["Date"])

# Save the cleaned dataset
df.to_csv("Final_Stock_Sentiment_Returns_Filled_Random_Cleaned.csv", index=False)
print("Data cleaning complete. Saved as Final_Stock_Sentiment_Returns_Filled_Random_Cleaned.csv")

# ==========================================
# Step 2: Central Limit Theorem (CLT) Verification
# ==========================================
print("\n=== Step 2: Central Limit Theorem (CLT) Verification ===")

sentiments = df["Sentiment_Score"].dropna()
sample_means = []

for _ in range(1000):
    sample = np.random.choice(sentiments, size=30, replace=True)
    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

plt.figure(figsize=(10,6))
plt.hist(sample_means, bins=30, density=True, alpha=0.6, color='g', label="Sample Means Histogram")
mu, sigma = norm.fit(sample_means)
x = np.linspace(min(sample_means), max(sample_means), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label="Fitted Normal Curve")
plt.title("CLT Verification: Distribution of Sample Means of Sentiment Score")
plt.xlabel("Sample Mean of Sentiment Score")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

stat, p_value = shapiro(sample_means)
print(f"Shapiro-Wilk Test p-value: {p_value}")

if p_value > 0.05:
    print("Sample mean distribution follows normality (fail to reject null hypothesis).")
else:
    print("Sample mean distribution does not follow normality.")

# ==========================================
# Step 3: Hypothesis Testing between High and Low Sentiment Groups
# ==========================================
print("\n=== Step 3: Hypothesis Testing between High and Low Sentiment Groups ===")

threshold = df["Sentiment_Score"].median()
high_sentiment = df[df["Sentiment_Score"] > threshold]["Return"]
low_sentiment  = df[df["Sentiment_Score"] <= threshold]["Return"]

print(f"High sentiment group size: {len(high_sentiment)}")
print(f"Low sentiment group size: {len(low_sentiment)}")

stat, p_value = mannwhitneyu(high_sentiment, low_sentiment, alternative="two-sided")
print(f"Mann-Whitney U Test p-value: {p_value}")

if p_value < 0.05:
    print("Significant return difference between high and low sentiment groups (reject null hypothesis).")
else:
    print("No significant return difference between high and low sentiment groups (fail to reject null hypothesis).")

stat, p_value = ks_2samp(high_sentiment, low_sentiment)
print(f"K-S Test p-value: {p_value}")

# ==========================================
# Step 4: Hypothesis Testing between Extreme Sentiment Groups
# ==========================================
print("\n=== Step 4: Hypothesis Testing between Extreme Sentiment Groups ===")

high_threshold = df["Sentiment_Score"].quantile(0.90)
low_threshold  = df["Sentiment_Score"].quantile(0.10)

extreme_high = df[df["Sentiment_Score"] >= high_threshold]["Return"]
extreme_low  = df[df["Sentiment_Score"] <= low_threshold]["Return"]

print(f"Extreme high sentiment group size: {len(extreme_high)}")
print(f"Extreme low sentiment group size: {len(extreme_low)}")

stat, p_value = mannwhitneyu(extreme_high, extreme_low, alternative="two-sided")
print(f"Extreme Sentiment Mann-Whitney U Test p-value: {p_value}")

if p_value < 0.05:
    print("Significant return difference between extreme sentiment groups (reject null hypothesis).")
else:
    print("No significant return difference between extreme sentiment groups (fail to reject null hypothesis).")

# ==========================================
# Step 5: Feature Engineering and Sequence Construction
# ==========================================
print("\n=== Step 5: Feature Engineering and Sequence Construction ===")

df = pd.read_csv("Final_Stock_Sentiment_Returns_Filled_Random_Cleaned.csv")
df = df[df["Stock"] == "AAPL"].sort_values("Date").reset_index(drop=True)

df["Next_Return"] = df["Return"].shift(-1)
df["Price"] = 100 * (1 + df["Return"]).cumprod()

df["SMA_5"]  = df["Price"].rolling(5).mean()
df["SMA_10"] = df["Price"].rolling(10).mean()
df["Momentum_5"] = df["Price"] - df["Price"].shift(5)

delta = df["Price"].diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100/(1+rs))

df = df.dropna(subset=["Next_Return", "SMA_5", "SMA_10", "Momentum_5", "RSI_14"]).reset_index(drop=True)

feature_cols = ["Sentiment_Score", "Return", "SMA_5", "SMA_10", "Momentum_5", "RSI_14"]
X_all = df[feature_cols].values
y_all = df["Next_Return"].values

# Sliding window sequence
timesteps = 5
X_seq, y_seq = [], []
for i in range(len(X_all) - timesteps):
    X_seq.append(X_all[i:i+timesteps])
    y_seq.append(y_all[i+timesteps])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("Sequence shapes:", X_seq.shape, y_seq.shape)

# Train/Validation/Test split and scaling
X_trval, X_test, y_trval, y_test = train_test_split(X_seq, y_seq, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.2, shuffle=False)

def scale_feats(X_train, X_val, X_test):
    n_tr, ts, nf = X_train.shape
    scaler = MinMaxScaler().fit(X_train.reshape(-1, nf))
    X_tr = scaler.transform(X_train.reshape(-1, nf)).reshape(n_tr, ts, nf)
    X_v  = scaler.transform(X_val.reshape(-1, nf)).reshape(X_val.shape)
    X_te = scaler.transform(X_test.reshape(-1, nf)).reshape(X_test.shape)
    return X_tr, X_v, X_te

X_train, X_val, X_test = scale_feats(X_train, X_val, X_test)

# ==========================================
# Step 6: Define Transformer Regression Model
# ==========================================
print("\n=== Step 6: Define and Train Transformer Regression Model ===")

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        pos = np.arange(seq_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2*(i//2)) / d_model)
        angle_rads = pos * angle_rates
        pos_enc = np.zeros((seq_len, d_model))
        pos_enc[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_enc[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.cast(pos_enc[None, ...], tf.float32)
        self.seq_len = seq_len
        self.d_model = d_model

    def call(self, x):
        return x + self.pos_encoding

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg

d_model  = 64
num_heads = 4
ff_dim = 128

inputs = Input(shape=(timesteps, X_train.shape[2]))
x = Dense(d_model)(inputs)
x = PositionalEncoding(timesteps, d_model)(x)

attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
x = LayerNormalization(epsilon=1e-6)(x + attn)

ff = Dense(ff_dim, activation="relu")(x)
ff = Dense(d_model)(ff)
x = LayerNormalization(epsilon=1e-6)(x + ff)

x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
outputs = Dense(1)(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.summary()

# ==========================================
# Step 7: Model Training
# ==========================================
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
cp = ModelCheckpoint("best_transformer_feats.h5", save_best_only=True, monitor="val_loss")

history = model.fit(
    X_train, y_train,
    epochs=200, batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[es, cp],
    verbose=2
)

# Plot training history
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.show()

# ==========================================
# Step 8: Model Evaluation
# ==========================================
print("\n=== Step 8: Model Evaluation ===")

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {loss:.5f}, MAE: {mae:.5f}")

y0 = np.zeros_like(y_test)
print("Baseline MSE:", np.mean((y_test - y0)**2),
      "Baseline MAE:", np.mean(np.abs(y_test - y0)))

y_pred = model.predict(X_val).flatten()

# Plot predictions
plt.figure(figsize=(10,5))
plt.plot(y_val, label="True", marker="o")
plt.plot(y_pred, label="Predicted", marker="x")
plt.legend(); plt.title("Validation: True vs Predicted"); plt.show()