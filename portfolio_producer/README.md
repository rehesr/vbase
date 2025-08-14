# Portfolio Producer

**Overview**  
This function creates market-cap-weighted stock portfolios that are based on sentiment and moving average sentiment change buckets. It pulls sentiment data from the Stocktwits API, merges it with S&P 500 constituent weights, applies bucketing rules, and outputs a clean, filtered portfolio.

**Features**  
- Fetches S&P 500 constituents and weights from local CSV files
- Retrieves and processes sentiment data from Stocktwits
    - Reads 30 minute sentiment data in the past week for each individual S&P 500 constituent, from https://api-gw-prd.stocktwits.com/api-middleware/external/sentiment/v2/{symbol}/chart
    - Resamples the data into 1 hour intervals, and calculates hourly sentiment change
    - Takes the moving average of sentiment change with window = 30  
- Classifies stocks into quantile-based buckets: bottom 25% = S(Small), middle 50% = M(Medium), top 25% = B(Big).   
- Selects stocks with specific sentiment bucket and moving average sentiment change bucket (these buckets need to be defined by the user in the input), and returns the portfolio as a DataFrame with symbols and weights  

**Setup**  
Make sure to place `constituents.csv` and `holdings-daily-us-en-spy.xlsx` in `data/stock_data/` directory.  

**Usage**  
Call `produce_portfolio` with sentiment and moving average sentiment change buckets (`"S"`, `"M"`, `"B"`, or None).  
Returns a DataFrame with:  
- **sym** — stock symbol  
- **wt** — portfolio weight  
This DataFrame represents the specific stocks and their weights in the portfolio, and can be applied directly to production.
