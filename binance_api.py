import requests
from datetime import datetime

base = "https://api.binance.com/api/v3"
symbo = "BTCUSDT"
base_currency = "USDT"
target_currency = "BTC"

def datetime_to_timestamp(datetime):
    return int(datetime.timestamp() * 1000)

def timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp / 1000)

def get_klines(start, end, interval="1m"):
    res = requests.get(
        f"{base}/klines",
        params={
            "symbol": symbo,
            "interval": interval,
            "startTime": datetime_to_timestamp(start)
            if isinstance(start, datetime)
            else start,
            "endTime": datetime_to_timestamp(end) if isinstance(end, datetime) else end,
            "limit": 1000,
        },
    )
    res.raise_for_status()
    return res.json()