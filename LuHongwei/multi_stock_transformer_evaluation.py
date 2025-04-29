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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 自定义 PositionalEncoding（和之前一样）
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
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
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.pos_encoding.shape[1],
                    "d_model": self.pos_encoding.shape[2]})
        return cfg

# 准备结果容器
results = []

# 假设 CSV 里就包含我们要的 10 支票，其名字存于这一列
df_all = pd.read_csv("Final_Stock_Sentiment_Returns_Filled_Random_Cleaned.csv")
stock_list = df_all["Stock"].unique()  # 若多于 10 支，可切片：[:10]

for stock in stock_list:
    df = df_all[df_all["Stock"] == stock].sort_values("Date").reset_index(drop=True)

    # 1. 构造标签和技术指标（与原来一致）
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

    df = df.dropna(subset=[
        "Next_Return","SMA_5","SMA_10",
        "Momentum_5","RSI_14"
    ]).reset_index(drop=True)

    # 2. 做成 sliding window 序列
    features = ["Sentiment_Score","Return","SMA_5","SMA_10","Momentum_5","RSI_14"]
    X_all = df[features].values
    y_all = df["Next_Return"].values

    ts = 5
    X_seq, y_seq = [], []
    for i in range(len(X_all) - ts):
        X_seq.append(X_all[i:i+ts])
        y_seq.append(y_all[i+ts])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # 3. 划分集
    X_trval, X_test, y_trval, y_test = train_test_split(
        X_seq, y_seq, test_size=0.15, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.2, shuffle=False
    )

    # 4. 归一化
    def scale_feats(X_tr, X_v, X_te):
        n, t, f = X_tr.shape
        flat = X_tr.reshape(-1, f)
        scaler = MinMaxScaler().fit(flat)
        X_tr_s = scaler.transform(flat).reshape(n, t, f)
        X_v_s  = scaler.transform(X_v.reshape(-1,f)).reshape(X_v.shape)
        X_te_s = scaler.transform(X_te.reshape(-1,f)).reshape(X_te.shape)
        return X_tr_s, X_v_s, X_te_s

    X_train, X_val, X_test = scale_feats(X_train, X_val, X_test)

    # 5. 构建 Transformer 回归模型
    d_model, num_heads, ff_dim = 64, 4, 128
    inp = Input(shape=(ts, X_train.shape[2]))
    x   = Dense(d_model)(inp)
    x   = PositionalEncoding(ts, d_model)(x)
    att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x   = LayerNormalization(epsilon=1e-6)(x + att)
    ff  = Dense(ff_dim, activation="relu")(x)
    ff  = Dense(d_model)(ff)
    x   = LayerNormalization(epsilon=1e-6)(x + ff)
    x   = GlobalAveragePooling1D()(x)
    x   = Dropout(0.3)(x)
    out = Dense(1)(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse", metrics=["mae"]
    )

    # 6. 训练
    es = EarlyStopping("val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, batch_size=32,
        callbacks=[es], verbose=0
    )

    # 7. 测试集上评估
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    # baseline：全零预测
    y0   = np.zeros_like(y_test)
    mse0 = np.mean((y_test - y0)**2)
    mae0 = np.mean(np.abs(y_test - y0))

    results.append({
        "Stock":       stock,
        "Test_MSE":    loss,
        "Test_MAE":    mae,
        "Baseline_MSE":mse0,
        "Baseline_MAE":mae0
    })

# 整理并显示所有股票的结果表
df_res = pd.DataFrame(results)
print(df_res)