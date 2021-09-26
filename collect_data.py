from binance_api import get_klines, timestamp_to_datetime
import pandas as pd
from datetime import datetime


def load_klines(from_date, to_date):
    while True:
        klines = get_klines(from_date, to_date)

        if not klines:
            break

        pd.DataFrame(
            {
                "open_time": [kline[0] for kline in klines],
                "close_time": [kline[6] for kline in klines],
                "low": [kline[3] for kline in klines],
                "high": [kline[2] for kline in klines],
                "open": [kline[1] for kline in klines],
                "close": [kline[4] for kline in klines],
                "volume": [kline[5] for kline in klines],
            }
        ).to_csv("data/klines.csv", mode="a", header=False, index=False)

        from_date = klines[-1][6]
        print(timestamp_to_datetime(klines[-1][0]))


def sync_kline_data():
    df = pd.read_csv("data/klines.csv")
    close_time = df["close_time"][len(df) - 1]
    load_klines(close_time, datetime.now())


# load_klines(datetime(2018, 9, 26), datetime.now()) # Initial data load
sync_kline_data()