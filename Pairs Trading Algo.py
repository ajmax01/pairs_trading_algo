import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime, timedelta

# Define pair stocks and return correlation, beta, spread for a time period

def obtain_training_data(tickers_y_on_x, training_start_date="2020-01-01", training_end_date="2026-08-05"):
    tix_data = yf.download(tickers=tickers_y_on_x, start=training_start_date, end=training_end_date).dropna().reindex(tickers_y_on_x, level=1,axis=1)
    x = sm.add_constant(tix_data["Close", tickers_y_on_x[1]])
    y = tix_data["Close", tickers_y_on_x[0]]
    model = sm.OLS(y, x)
    results = model.fit()
    returns = np.log(tix_data["Close"] / tix_data["Close"].shift(1)).dropna()

    constant = results.params["const"]
    beta = results.params.iloc[1]
    correlation = returns.corr().iloc[1,0]

    spread = y - (constant + beta * tix_data["Close", tickers_y_on_x[1]])
    training_spread_mean = spread.mean()
    training_spread_stddev = spread.std()

    return constant, beta, training_spread_mean, training_spread_stddev 

# Backtest

def backtest(ticker_pair, start_date="2020-01-01", end_date="2023-01-01", training_window_length=10):
    start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    training_start_date_datetime = start_date_datetime - timedelta(days=training_window_length*365)
    training_start_date_string = datetime.strftime(training_start_date_datetime, "%Y-%m-%d")

    constant, beta, mean, stddev = obtain_training_data(tickers_y_on_x=ticker_pair, training_start_date=training_start_date_string, training_end_date=start_date)
    ticker_y = ticker_pair[0]
    ticker_x = ticker_pair[1]
    
    orders_list = []

    history = yf.download(tickers= ticker_pair, start=start_date, end=end_date).reindex(ticker_pair, level=1, axis=1)
    spread = history["Close", ticker_y] - (constant + beta * history["Close", ticker_x])
    spread_z_score = (spread) / stddev

    history["z of spread"] = spread_z_score

    z_array = history["z of spread"].to_numpy()
    date_array = history.index.to_numpy()
    close_price_array = history["Close"].to_numpy()

    # Algo Logic
    long_x_short_y = False
    long_y_short_x = False
    portfolio_value = 100000
    for i in range(len(history)):
        if z_array[i] > 1 and not long_x_short_y: 
            orders_list.append({"Date": date_array[i], "Stock": ticker_x, "Type": "Buy", "Value":portfolio_value/3, "Price": close_price_array[i, 1],
                                 "Profit": None})
            orders_list.append({"Date": date_array[i], "Stock": ticker_y, "Type": "Sell Short", "Value":portfolio_value/3, "Price": close_price_array[i, 0],
                                "Profit": None})
            long_x_short_y = True
        elif z_array[i] < 0.5 and long_x_short_y:
            x_long_close_value = (((close_price_array[i, 1] - orders_list[-2]["Price"])/(orders_list[-2]["Price"]))+1)*(orders_list[-2]["Value"])
            y_short_close_value = (((close_price_array[i, 0] - orders_list[-1]["Price"])/(orders_list[-1]["Price"]))-1)*(orders_list[-1]["Value"])*-1
            portfolio_value = x_long_close_value + y_short_close_value + (portfolio_value - orders_list[-2]["Value"] - orders_list[-1]["Value"])
            orders_list.append({"Date": date_array[i], "Stock": ticker_x, "Type": "Sell", "Value":x_long_close_value, "Price": close_price_array[i, 1], 
                                "Profit": None})
            orders_list.append({"Date": date_array[i], "Stock": ticker_y, "Type": "Buy to Cover", "Value":y_short_close_value, "Price": close_price_array[i, 0],
                                 "Profit": None})
            long_x_short_y = False
        elif z_array[i] < -1 and not long_y_short_x: 
            orders_list.append({"Date": date_array[i], "Stock": ticker_y, "Type": "Buy", "Value":portfolio_value/3, "Price": close_price_array[i, 0],
                                 "Profit": None})
            orders_list.append({"Date": date_array[i], "Stock": ticker_x, "Type": "Sell Short", "Value":portfolio_value/3, "Price": close_price_array[i, 1],
                                 "Profit": None})
            long_y_short_x = True
        elif z_array[i] > -0.5 and long_y_short_x: 
            y_long_close_value = (((close_price_array[i, 0] - orders_list[-2]["Price"])/(orders_list[-2]["Price"]))+1)*(orders_list[-2]["Value"])
            x_short_close_value = (((close_price_array[i, 1] - orders_list[-1]["Price"])/(orders_list[-1]["Price"]))-1)*(orders_list[-1]["Value"])*-1
            portfolio_value = x_short_close_value + y_long_close_value + (portfolio_value - orders_list[-2]["Value"] - orders_list[-1]["Value"])
            orders_list.append({"Date": date_array[i], "Stock": ticker_y, "Type": "Sell", "Value":y_long_close_value, "Price": close_price_array[i, 0],
                                 "Profit": None})
            orders_list.append({"Date": date_array[i], "Stock": ticker_x, "Type": "Buy to Cover", "Value": x_short_close_value, "Price": close_price_array[i, 1],
                                 "Profit": None})
            long_y_short_x = False

    
    orders_df = pd.DataFrame(orders_list)

    return orders_df, portfolio_value

orders, portfolio_value = backtest(["PL=F", "PA=F"], start_date="2023-06-01", end_date = "2024-6-30")
print(orders,"\n",portfolio_value)




