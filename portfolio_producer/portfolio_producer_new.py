"""
This module contains the logic for producing portfolio positions.
"""

from datetime import datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from datetime import time
import numpy as np

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

os.environ["DATA_DIR"] = "data"

try:
    DATA_DIR = Path(os.environ["DATA_DIR"])
except KeyError:
    raise EnvironmentError("DATA_DIR environment variable is not set.")

sentiment_file = DATA_DIR / "Stocktwits"/ "sentiment"
weight_file = DATA_DIR / "stock_data"



def produce_portfolio(portfolio_date: str, logger: object, st_bucket=None, ma_sc_bucket=None) -> pd.DataFrame:
    """
    Produces the market cap weighted portfolio with S and B buckets for sentiment and moving average sentiment change.

    Args:
        portfolio_date: The date for which to produce the portfolio in YYYY-MM-DD format.

    Returns:
        A DataFrame containing the position with columns ['time', 'sym', 'wt'].

    """
    portfolio_datetime = pd.to_datetime(portfolio_date,unit='ns', utc=True)
    current_datetime = pd.to_datetime(datetime.now(),unit='ns', utc=True)
    if current_datetime >= pd.to_datetime('2025-06-01'):
        current_datetime = pd.to_datetime('2025-05-31',unit='ns', utc=True)

    # Fetch market weights 
    constituents = pd.read_csv(weight_file / "constituents.csv")
    weights_df = pd.read_excel(weight_file / "holdings-daily-us-en-spy.xlsx", skiprows=4, skipfooter=10)

    constituents = constituents[['Symbol','GICS Sector','GICS Sub-Industry']]
    weights_df = weights_df[['Name','Ticker','Weight']]
    weights_df.rename(columns={'Ticker':'Symbol'},inplace=True)

    spy_cons = pd.merge(weights_df,constituents,on=['Symbol'])
    spy_cons.rename(columns={'Symbol':'symbol'},inplace=True)

    spy_cons = spy_cons[['symbol','GICS Sector','Weight']]
    cons = spy_cons['symbol']

        
    # Fetch feature data
    all_factors = pd.DataFrame()
    freq = '1h'
    months = ['0'+str(i) if i < 10 else str(i) for i in range(1,13)]
    # years = ['2023','2024','2025']
    years = ['2025']
    for y in years:
        for m in months:
            if y == '2025' and m >= '06':
                continue
            else:
                st = pd.read_csv(sentiment_file / f"sentiment_{y}_{m}.csv")
                st = st[st['symbol'].isin(cons)]
                st.rename(columns={'created_at':'time'},inplace=True)
                st['normalized_sentiment'] = st.groupby('time')['sentiment'].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)

                st['time'] = pd.to_datetime(st['time'],unit='ns', utc=True)
                st.set_index('time',inplace=True)
                st = st[st['symbol'].isin(cons)]
                st = st[st.index.time>=time(9, 30)]
                st = st[st.index.time<=time(16, 0)]
                st = st.groupby('symbol').resample(freq, offset='30min').last().drop('symbol',axis=1).reset_index()
                
                st['sentiment_change'] = st.groupby('symbol')['sentiment'].transform(
                    lambda x: np.where(x.shift(1) != 0, x/x.shift(1) - 1, 0)
                )
                st['normalized_sentiment_change'] = st.groupby('time')['sentiment_change'].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
                st.set_index('time',inplace=True)
                st = st[['symbol','sentiment','normalized_sentiment','sentiment_change','normalized_sentiment_change']]
                all_factors = pd.concat([all_factors,st])

    all_factors = all_factors[all_factors['symbol'].isin(cons)]
    all_factors = all_factors[(all_factors.index >= portfolio_datetime) & (all_factors.index <= current_datetime)]
    all_factors = all_factors[all_factors.index.time>=time(9, 30)]
    all_factors = all_factors[all_factors.index.time<=time(16, 0)]


    def qcut_safe(x):
        return pd.qcut(
            x,
            q=[0, 0.25, 0.75, 1],
            labels=False,  # Returns 0, 1, 2 instead of 'S', 'M', 'B'
            duplicates='drop'
        )

    # Later, map indices to labels if needed
    bin_map = {0: 'S', 1: 'M', 2: 'B'}
    all_factors['st_bucket'] = all_factors.groupby('time')['normalized_sentiment'].transform(qcut_safe).map(bin_map)

    all_factors['time'] = all_factors.index
    all_factors = pd.merge(all_factors,spy_cons,on='symbol')
    all_factors['ma_normalized_sentiment_change'] = list(all_factors.groupby('time')['normalized_sentiment_change'].rolling(30).mean())
    all_factors_new = all_factors.dropna()

    all_factors_new['ma_sc_bucket'] = all_factors_new.groupby('time')['ma_normalized_sentiment_change'].transform(qcut_safe).map(bin_map)


    df = all_factors_new.copy()
    df.rename(columns={'symbol':'sym'},inplace=True)

    if st_bucket is not None:
        df = df[df["st_bucket"] == st_bucket]
    if ma_sc_bucket is not None:
        df = df[df["ma_sc_bucket"] == ma_sc_bucket]

    if df.empty:
        raise ValueError("No data available for the specified time and bucket(s).")

    df['wt'] = df.groupby('time')['Weight'].transform(lambda x: x / x.sum())

    # Final portfolio DataFrame
    position_df = df[['time', 'sym', 'wt']].sort_values(['time', 'wt'], ascending=[True, False]).reset_index(drop=True)
    
    return position_df