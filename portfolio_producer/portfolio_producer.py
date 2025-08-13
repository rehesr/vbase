"""
This module contains the logic for producing portfolio positions.
"""

from datetime import datetime, timedelta
import cloudscraper
import pandas as pd
import pandas_market_calendars as mcal
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ["DATA_DIR"] = "data"

try:
    DATA_DIR = Path(os.environ["DATA_DIR"])
except KeyError:
    raise EnvironmentError("DATA_DIR environment variable is not set.")

weight_file = DATA_DIR / "stock_data"


def produce_portfolio(st_bucket="S", ma_sc_bucket="B") -> pd.DataFrame:
    """
    Produces the market cap weighted portfolio with S and B buckets for sentiment and moving average sentiment change.

    Args:
        portfolio_date: The date for which to produce the portfolio in YYYY-MM-DD format.

    Returns:
        A DataFrame containing the position with columns ['time', 'sym', 'wt'].

    """
    # Fetch market weights
    constituents = pd.read_csv(weight_file / "constituents.csv")
    weights_df = pd.read_excel(weight_file / "holdings-daily-us-en-spy.xlsx", skiprows=4, skipfooter=10)

    constituents = constituents[['Symbol','GICS Sector','GICS Sub-Industry']]
    weights_df = weights_df[['Name','Ticker','Weight']]
    weights_df.rename(columns={'Ticker':'Symbol'},inplace=True)

    spy_cons = pd.merge(weights_df,constituents,on=['Symbol'])
    spy_cons.rename(columns={'Symbol':'symbol'},inplace=True)

    spy_cons = spy_cons[['symbol','GICS Sector','Weight']]
    spy_tickers = spy_cons['symbol']

    user = "vbasedan"
    password = "!BB7X$qUea"
    zoom = "1W"

    scraper = cloudscraper.create_scraper()

    rows = []

    for symbol in spy_tickers:
        url = f"https://api-gw-prd.stocktwits.com/api-middleware/external/sentiment/v2/{symbol}/chart"

        try:
            response = scraper.get(url, auth=(user, password) ,params={"zoom": zoom})
            if response.status_code != 200:
                print(f"[{symbol}] Failed: {response.status_code}")
                continue

            data = response.json()
            raw_data = data['data']  
            if not raw_data:
                print(f"[{symbol}] No sentiment data.")
                continue

            df = pd.DataFrame.from_dict(raw_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            df = df.resample('1h').last()[['sentimentNormalized']]
            df.fillna(method='ffill', inplace=True)
            df['sentiment_change'] = df['sentimentNormalized'] / df['sentimentNormalized'].shift(1) - 1
            df['ma_sentiment_change'] = df['sentiment_change'].rolling(window=30).mean()
            df['symbol'] = symbol

            # Add latest row
            latest_row = df.iloc[-1]
            rows.append(latest_row)

        except Exception as e:
            print(f"[{symbol}] Error: {e}")

        time.sleep(0.5)  

    # Factors dataframe
    all_factors = pd.DataFrame(rows)
    all_factors = all_factors[['symbol', 'sentimentNormalized', 'ma_sentiment_change']]
    

    def qcut_safe(x):
            return pd.qcut(
            x,
            q=[0, 0.25, 0.75, 1],
            labels=['S', 'M', 'B'],
            duplicates='drop'
        )

    
    all_factors['st_bucket'] = qcut_safe(all_factors['sentimentNormalized'])
    all_factors['ma_sc_bucket'] = qcut_safe(all_factors['ma_sentiment_change'])
    all_factors = pd.merge(all_factors,spy_cons,on='symbol')

    df = all_factors.copy()
    df.rename(columns={'symbol':'sym','Weight':'wt'},inplace=True)

    if st_bucket is not None:
        df = df[df["st_bucket"] == st_bucket]
    if ma_sc_bucket is not None:
        df = df[df["ma_sc_bucket"] == ma_sc_bucket]

    if df.empty:
        raise ValueError("No data available for the specified time and bucket(s).")
    
    position_df = df[['sym', 'wt']].sort_values(['wt'], ascending=[False]).reset_index(drop=True)
    return position_df