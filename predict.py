import tensorflow as tf
import pandas as pd
from constants import SEQ_LEN
import numpy as np
import joblib
from binance_api import timestamp_to_datetime
from sklearn.preprocessing import MinMaxScaler

model_number = "2021_09_26_23_09"
model_dir = f"training/{model_number}/"

df = pd.read_csv("data/klines.csv")

# Prep sequence data
last_row_index = len(df) - 1
seq_data = []
for seq_index in range(SEQ_LEN):
    trade_row = []
    for mins in (5, 60, 1440):
        end_index = last_row_index - mins * seq_index
        start_index = end_index - mins + 1
        trade_row.append(min(df["low"][start_index : end_index + 1]))
        trade_row.append(max(df["high"][start_index : end_index + 1]))
        trade_row.append(df["open"][start_index])
        trade_row.append(df["close"][end_index])
        trade_row.append(sum(df["volume"][start_index : end_index + 1]))
    seq_data.append(trade_row)
seq_data.reverse()

seq_data = np.array(seq_data)

# Normalize data
columns = [
    "low_5",
    "high_5",
    "open_5",
    "close_5",
    "volume_5",
    "low_60",
    "high_60",
    "open_60",
    "close_60",
    "volume_60",
    "low_1440",
    "high_1440",
    "open_1440",
    "close_1440",
    "volume_1440",
]

for index, column in enumerate(columns):
    # scaler = joblib.load(f"{model_dir}scalers/{column}.gz")
    scaler = MinMaxScaler()
    seq_data[:, index : index + 1] = scaler.fit_transform(
        seq_data[:, index : index + 1]
    )

# Predict
checkpoint_path = f"{model_dir}checkpoint"
model = tf.keras.models.load_model(checkpoint_path)
seq_data = np.reshape(seq_data, (-1, seq_data.shape[0], seq_data.shape[1]))

print(timestamp_to_datetime(df['close_time'][last_row_index]))
print(model.predict(seq_data))
print("0: won't rise rate, 1: will rise rate")
