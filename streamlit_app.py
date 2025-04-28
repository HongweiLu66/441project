import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D
)
import matplotlib.pyplot as plt

# ========================================
# Positional Encoding Layer
# ========================================
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        pos = np.arange(seq_len)[:, None]
        i   = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2*(i//2))/d_model)
        angle_rads  = pos * angle_rates
        pos_enc     = np.zeros((seq_len, d_model))
        pos_enc[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_enc[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.cast(pos_enc[None, ...], tf.float32)

    def call(self, x):
        return x + self.pos_encoding

# ========================================
# Build Transformer Model
# ========================================
def build_model(timesteps, num_features):
    d_model  = 64
    num_heads= 4
    ff_dim   = 128

    inputs = Input(shape=(timesteps, num_features))
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
    return model

# ========================================
# Streamlit app starts here
# ========================================

def main():
    st.title("Stock Return Prediction (Rebuild Transformer + Load Weights)")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        stocks = df["Stock"].unique()
        stock_choice = st.selectbox("Choose a stock", stocks)

        if st.button("Start Prediction"):
            run_model(df, stock_choice)

def run_model(df, stock_choice):
    df = df[df["Stock"]==stock_choice].sort_values("Date").reset_index(drop=True)
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

    df = df.dropna(subset=["Next_Return","SMA_5","SMA_10","Momentum_5","RSI_14"]).reset_index(drop=True)

    feature_cols = ["Sentiment_Score","Return","SMA_5","SMA_10","Momentum_5","RSI_14"]
    X_all = df[feature_cols].values
    y_all = df["Next_Return"].values

    # Build sequences
    timesteps = 5
    X_seq, y_seq = [], []
    for i in range(len(X_all)-timesteps):
        X_seq.append(X_all[i:i+timesteps])
        y_seq.append(y_all[i+timesteps])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Split
    X_trval, X_test, y_trval, y_test = train_test_split(X_seq, y_seq, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.2, shuffle=False)

    # Scale
    X_train, X_val, X_test = scale_feats(X_train, X_val, X_test)

    # Build and load model
    model = build_model(timesteps, X_train.shape[2])
    model.load_weights("best_transformer_feats.h5")

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Test MSE: {loss:.5f}, Test MAE: {mae:.5f}")

    # Predict
    y_pred = model.predict(X_val).flatten()
    st.subheader("Validation Set: True vs Predicted")
    fig2, ax2 = plt.subplots()
    ax2.plot(y_val, label="True", marker="o")
    ax2.plot(y_pred, label="Pred", marker="x")
    ax2.legend()
    st.pyplot(fig2)

def scale_feats(X_train, X_val, X_test):
    n_tr, ts, nf = X_train.shape
    flat = X_train.reshape(-1, nf)
    scaler = MinMaxScaler().fit(flat)
    X_tr = scaler.transform(flat).reshape(n_tr, ts, nf)
    X_v  = scaler.transform(X_val.reshape(-1, nf)).reshape(X_val.shape)
    X_te = scaler.transform(X_test.reshape(-1, nf)).reshape(X_test.shape)
    return X_tr, X_v, X_te

if __name__ == "__main__":
    main()
