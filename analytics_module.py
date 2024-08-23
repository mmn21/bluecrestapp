#!/usr/bin/env python
# coding: utf-8



import sqlite3
import pandas as pd
import numpy as np
np.bool = np.bool_
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression



import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



### GLOBAL VARIABLES ###

db_name = 'stocks_data.db'




### HELPER FUNCTIONS ###

def fetch_tables_as_dataframe(db_name):
    """
    Fetches a table from a SQLite database and converts it to a Pandas DataFrame.
    
    Parameters:
    db_name (str): Name of the SQLite database file (e.g., 'my_database.db').
    table_name (str): Name of the table to fetch.
    
    Returns:
    pd.DataFrame: DataFrame containing the data from the specified table.
    """
    # Establish a connection to the SQLite database
    conn = sqlite3.connect(db_name)
    
    # Query to fetch stock tickers
    tickers_query = '''
        SELECT stock_id, ticker, company_name FROM Stocks
    '''
    
    # Filter out duplicate tickers when a company has different classes of shares
    tickers_df = pd.read_sql(tickers_query, conn)
    tickers_df = tickers_df.drop_duplicates(subset='company_name', keep='first', inplace=False, ignore_index=False)
    tickersdf = tickers_df.drop(columns = ['company_name'])
    
    # Query to fetch price data
    price_data_query = '''
        SELECT stock_id, price_date, adjusted_close_price FROM DailyPrices
    '''
    prices_df = pd.read_sql(price_data_query, conn)

    # Close the connection
    conn.close()
    
    # Pivot the price data so that the columns are the tickers and rows are the dates
    prices_df = prices_df.merge(tickers_df, on='stock_id')
    pivot_df = prices_df.pivot(index='price_date', columns='ticker', values='adjusted_close_price')

    pivot_df.index = pd.to_datetime(pivot_df.index)
    return pivot_df

def calc_correlations(df,start_date, end_date, display = 10):
    # Filter DataFrame by date range
    filter_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
   
    # Calculate the log returns
    log_returns = np.log(filter_df / df.shift(1))
    
    # Compute the correlation matrix
    corr_matrix = round(log_returns.corr(),3)
    
    # Extract the upper triangle of the correlation matrix without the diagonal
    corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Stack and sort the correlation pairs
    corr_pairs = corr_pairs.stack()
    corr_pairs.index.names = ['Stock 1', 'Stock 2']
    corr_pairs = corr_pairs.reset_index()
    corr_pairs.columns = ['Stock 1', 'Stock 2', 'Correlation']

    top_corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False).head(display).reset_index(drop = True)
    
    return top_corr_pairs

def calculate_rolling_correlation(df, asset1, asset2, window):
    """
    Calculate the rolling correlation between two assets over a specified window.

    Parameters:
    df (pd.DataFrame): DataFrame containing the price data for the assets. Each column should represent an asset.
    asset1 (str): The name of the first asset (column name in the DataFrame).
    asset2 (str): The name of the second asset (column name in the DataFrame).
    window (int): The rolling window size (number of periods).

    Returns:
    pd.Series: A series containing the rolling correlation between the two assets.
    """
    # Check if the specified assets exist in the DataFrame
    if asset1 not in df.columns or asset2 not in df.columns:
        raise ValueError("One or both of the specified assets do not exist in the DataFrame.")
    
    # Calculate the rolling correlation
    
    log_returns = np.log(df / df.shift(1))

    rolling_corr = log_returns[asset1].rolling(window=window).corr(log_returns[asset2])
    
    return rolling_corr


def extract_asset_pairs(df):
    """
    Extracts a list of asset pairs from adjacent cells in columns 'stock 1' and 'stock 2' of the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['stock 1', 'stock 2', 'correlation']

    Returns:
    list: List of tuples where each tuple is a pair of assets.
    """
    pairs = []
    
    for _, row in df.iterrows():
        stock1 = row['Stock 1']
        stock2 = row['Stock 2']
        pairs.append((stock1, stock2))
    
    return pairs

def calculate_cointegration_for_pairs(asset_pairs, price_df, start_date, end_date):
    """
    Calculate the cointegration for a list of asset pairs within a specified date range.

    Parameters:
    asset_pairs (list): List of tuples, where each tuple contains a pair of asset tickers (e.g., [('AAPL', 'MSFT'), ('GOOG', 'AMZN')]).
    price_df (pd.DataFrame): DataFrame containing daily price data for all assets. Columns should be asset tickers, and the index should be dates.
    start_date (str): Start date for filtering the DataFrame (format: 'YYYY-MM-DD').
    end_date (str): End date for filtering the DataFrame (format: 'YYYY-MM-DD').

    Returns:
    pd.DataFrame: DataFrame containing the asset pairs and their cointegration p-values.
    """
    # Filter the DataFrame by the specified date range
    filtered_df = price_df.loc[start_date:end_date]

    results = []

    # Iterate over the list of asset pairs
    for asset1, asset2 in asset_pairs:
        # Extract the price data for the two assets
        prices1 = filtered_df[asset1]
        prices2 = filtered_df[asset2]

        # Calculate the cointegration p-value
        p_value = calculate_cointegration(prices1, prices2)
        results.append((asset1, asset2, p_value))

    # Convert the results into a DataFrame for easier handling
    coint_df = pd.DataFrame(results, columns=['Stock 1', 'Stock 2', 'Cointegration P-Value'])

    return coint_df

def calculate_cointegration(prices1, prices2):
    """
    Calculate the cointegration p-value between two asset price series.

    Parameters:
    prices1 (pd.Series): Time series of the first asset's prices.
    prices2 (pd.Series): Time series of the second asset's prices.

    Returns:
    float: p-value of the cointegration test.
    """
    score, p_value, _ = coint(prices1, prices2)
    return round(p_value,3)

def calculate_vol_correlation(price_series1, price_series2, window=20):
    """
    Calculate the correlation of the volatility between two asset price series.
    Handles NaN and inf values by only using rows where both series have valid values.
    
    Parameters:
    - price_series1: pd.Series or np.array, time series of asset 1 prices
    - price_series2: pd.Series or np.array, time series of asset 2 prices
    - window: int, the rolling window size for calculating volatility (default is 20 days)

    Returns:
    - correlation: float, the correlation between the volatilities of the two assets
    """
    # Convert to pandas Series if necessary
    price_series1 = pd.Series(price_series1)
    price_series2 = pd.Series(price_series2)
    
    # Drop rows with NaN or inf values
    valid_mask = price_series1.notna() & price_series2.notna()
    valid_mask &= np.isfinite(price_series1) & np.isfinite(price_series2)
    
    price_series1 = price_series1[valid_mask]
    price_series2 = price_series2[valid_mask]
    
    # Ensure there is data left after cleaning
    if len(price_series1) < window:
        return 0
    
    # Calculate returns
    returns1 = price_series1.pct_change().dropna()
    returns2 = price_series2.pct_change().dropna()
    
    # Calculate rolling volatility (standard deviation of returns)
    rolling_volatility1 = returns1.rolling(window=window).std().dropna()
    rolling_volatility2 = returns2.rolling(window=window).std().dropna()
    
    # Drop rows where either volatility series has NaN values
    valid_mask = rolling_volatility1.notna() & rolling_volatility2.notna()
    rolling_volatility1 = rolling_volatility1[valid_mask]
    rolling_volatility2 = rolling_volatility2[valid_mask]
    
    # Ensure there is data left after cleaning
    if len(rolling_volatility1) == 0:
        return 0
    
    # Calculate correlation between volatilities
    correlation = np.corrcoef(rolling_volatility1, rolling_volatility2)[0, 1]
    
    return round(correlation,3)

def calc_vol_corr_pairs(asset_pairs, price_df, start_date, end_date):
    """
    Calculate the correlation of the volaility of a list of assets 

    Parameters:
    asset_pairs (list): List of tuples, where each tuple contains a pair of asset tickers (e.g., [('AAPL', 'MSFT'), ('GOOG', 'AMZN')]).
    price_df (pd.DataFrame): DataFrame containing daily price data for all assets. Columns should be asset tickers, and the index should be dates.
    start_date (str): Start date for filtering the DataFrame (format: 'YYYY-MM-DD').
    end_date (str): End date for filtering the DataFrame (format: 'YYYY-MM-DD').

    Returns:
    pd.DataFrame: DataFrame containing the asset pairs and the correlation values of their volatilities.
    """
    # Filter the DataFrame by the specified date range
    filtered_df = price_df.loc[start_date:end_date]

    results = []

    # Iterate over the list of asset pairs
    for asset1, asset2 in asset_pairs:
        # Extract the price data for the two assets
        prices1 = filtered_df[asset1]
        prices2 = filtered_df[asset2]

        # Calculate the cointegration p-value
        corr = calculate_vol_correlation(prices1, prices2, window=20)
        results.append((asset1, asset2, corr))

    # Convert the results into a DataFrame for easier handling
    corr_df = pd.DataFrame(results, columns=['Stock 1', 'Stock 2', 'Vol Correlation'])

    return corr_df

#### STRATEGY BACKTESTING FUNCTIONS #########

def calculate_z_score(series: pd.Series, window: int) -> pd.Series:
    """Calculate the Z-score for a given series."""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    z_score = (series - mean) / std
    return z_score

def simulate_pairs_trading_strategy(prices: pd.DataFrame, ticker1: str, ticker2: str, z_threshold: float,z_exit: float, z_window: int) -> pd.DataFrame:
    """
    Simulates a pairs trading strategy based on Z-score.

    Parameters:
    prices (pd.DataFrame): DataFrame containing price data for both tickers.
    ticker1 (str): The first stock ticker.
    ticker2 (str): The second stock ticker.
    z_threshold (float): The Z-score threshold to trigger trades.
    z_window (int): The window size to calculate the Z-score.

    Returns:
    pd.DataFrame: A DataFrame with the signals, positions, and P&L.
    """
    # Calculate the price ratio and its Z-score
    ratio = prices[ticker1] / prices[ticker2]
    z_scores = calculate_z_score(ratio, z_window)

    # Initialize position and P&L tracking
    position = 0  # 1 for long, -1 for short, 0 for no position
    entry_price1 = 0
    entry_price2 = 0
    results = []

    for i in range(z_window, len(prices)):
        # Entry conditions
        if position == 0:
            if z_scores[i] > z_threshold:
                # Enter short position
                position = -1
                entry_price1 = prices[ticker1].iloc[i]
                entry_price2 = prices[ticker2].iloc[i]
                results.append((prices.index[i], ticker1, ticker2, 'Short', entry_price1, entry_price2, 0))
            elif z_scores[i] < -z_threshold:
                # Enter long position
                position = 1
                entry_price1 = prices[ticker1].iloc[i]
                entry_price2 = prices[ticker2].iloc[i]
                results.append((prices.index[i], ticker1, ticker2, 'Long', entry_price1, entry_price2, 0))

        # Exit conditions
        elif position != 0:
            if (position == 1 and z_scores[i] >= -1*z_exit) or (position == -1 and z_scores[i] <= z_exit):
                exit_price1 = prices[ticker1].iloc[i]
                exit_price2 = prices[ticker2].iloc[i]
                pnl = 100*((exit_price1/exit_price2) - (entry_price1/entry_price2))/(entry_price1/entry_price2) * position
                results[-1] = (prices.index[i], ticker1, ticker2, 'Close', entry_price1/entry_price2, exit_price1/exit_price2, pnl)
                position = 0

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=['Date', 'Ticker1', 'Ticker2', 'Action', 'Entry_Price', 'Exit_Price', 'PnL%'])
    return results_df


######## RSI STRATEGY HELPER FUNCTIONS ##############

def calculate_rsi(series, window):
    """Calculate the Relative Strength Index (RSI) for a given series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def simulate_rsi_trading_strategy(df, ticker1, ticker2, upper_rsi_threshold, lower_rsi_threshold, rsi_exit_threshold, rsi_window):
    """
    Simulate a trading strategy based on RSI.
    
    Parameters:
    - df: DataFrame containing price data with columns for ticker1 and ticker2.
    - ticker1: First stock ticker.
    - ticker2: Second stock ticker.
    - upper_rsi_threshold: RSI value above which to enter a short trade.
    - lower_rsi_threshold: RSI value below which to enter a long trade.
    - rsi_exit_threshold: RSI value to close trades.
    - rsi_window: Window size for RSI calculation.
    
    Returns:
    - trades: List of trades with entry and exit details.
    - portfolio: DataFrame showing the portfolio value over time.
    """
    
    # Calculate the ratio of the two tickers
    df['ratio'] = df[ticker1] / df[ticker2]
    
    # Calculate the RSI of the ratio
    df['rsi'] = calculate_rsi(df['ratio'], rsi_window)
    
    # Initialize variables
    position = None
    entry_price = 0
    trades = []
    
    # Iterate through the DataFrame to simulate the strategy
    for i in range(1, len(df)):
        rsi = df['rsi'].iloc[i]
        
        # Check for entry signals
        if position is None:
            if rsi > upper_rsi_threshold:
                # Enter short trade
                position = 'short'
                entry_price = df['ratio'].iloc[i]
                trades.append({'type': 'short', 'entry_date': df.index[i], 'entry_price': entry_price})
            elif rsi < lower_rsi_threshold:
                # Enter long trade
                position = 'long'
                entry_price = df['ratio'].iloc[i]
                trades.append({'type': 'long', 'entry_date': df.index[i], 'entry_price': entry_price})
        
        # Check for exit signals
        elif position == 'short' and rsi <= rsi_exit_threshold:
            # Exit short trade
            exit_price = df['ratio'].iloc[i]
            trades[-1].update({'exit_date': df.index[i], 'exit_price': exit_price, 'profit': 100*(entry_price - exit_price)/entry_price})
            position = None
            
        elif position == 'long' and rsi >= rsi_exit_threshold:
            # Exit long trade
            exit_price = df['ratio'].iloc[i]
            trades[-1].update({'exit_date': df.index[i], 'exit_price': exit_price, 'profit': 100*(exit_price - entry_price)/entry_price})
            position = None
    
    trades_df = pd.DataFrame(trades)
    trades_df.columns = ['Action','Entry Date','Entry Price','Exit Date','Exit Price','PnL%']
    
    return trades_df




#### STRATEGY METRICS HELPER FUNCTIONS ####

def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0) -> float:
    """Calculate the Sharpe Ratio."""
    years = ((price_data.index[-1] - price_data.index[0]).days)/365
    periods_per_year = len(returns)/years
    excess_returns = np.array(returns) - risk_free_rate / periods_per_year
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)

def calculate_sortino_ratio(returns: list, risk_free_rate: float = 0) -> float:
    """Calculate the Sortino Ratio."""
    years = ((price_data.index[-1] - price_data.index[0]).days)/365
    periods_per_year = len(returns)/years
    excess_returns = np.array(returns) - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(periods_per_year)

def calculate_cagr(returns: list) -> float:
    """Calculate the Compound Annual Growth Rate (CAGR)."""
    cumulative_return = np.sum(returns)
    n_periods = len(returns)
    years = ((price_data.index[-1] - price_data.index[0]).days)/365
    periods_per_year = len(returns)/years
    return (cumulative_return/100)**(periods_per_year / n_periods) 

def calculate_max_drawdown(returns: list) -> float:
    """Calculate the Maximum Drawdown."""
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns - running_max
    return np.min(drawdowns)

def calculate_calmar_ratio(returns: list) -> float:
    """Calculate the Calmar Ratio."""
    cagr = calculate_cagr(returns)
    max_drawdown = calculate_max_drawdown(returns)
    return cagr / abs(max_drawdown)

def calculate_metrics(returns: list, risk_free_rate: float = 0) -> pd.DataFrame:
    """
    Calculate and return a table of key trading strategy metrics.

    Parameters:
    returns (list): A list of returns from a trading strategy.
    risk_free_rate (float): The risk-free rate for Sharpe and Sortino ratio calculations.
    periods_per_year (int): Number of trading periods in a year.

    Returns:
    pd.DataFrame: A DataFrame with metric names and their corresponding values.
    """
    total_return = np.sum(returns)
    years = ((price_data.index[-1] - price_data.index[0]).days)/365
    periods_per_year = len(returns)/years
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)
    calmar_ratio = calculate_calmar_ratio(returns)
    max_drawdown = calculate_max_drawdown(returns)

    metrics = {
        'Metric': ['Total Return', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown'],
        'Value': [total_return, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown]
    }

    return pd.DataFrame(metrics).round(3)

def parameter_performance(ticker1,ticker2,rolling_window, metric):
    z_entry =np.round(np.arange(0, 4, 0.1),1)  # Z Score Range for entry
    z_exit = np.round(np.arange(0, 2.5, 0.10),1)    # Z Score Range for exit
    
    heatmap_data = pd.DataFrame(index=z_exit, columns=z_entry)

    for x in z_entry:
        for y in z_exit:
            try:
                heatmap_data.loc[y,x] = calculate_metrics(simulate_pairs_trading_strategy(price_data, ticker1,ticker2, x ,y,rolling_window)['PnL%']).set_index('Metric',drop = True).T.reset_index(drop = True)[metric][0]
            except:
                heatmap_data.loc[y,x] = 0

    return heatmap_data

def optimal_conditions_table(perf_df,window,metric):
    '''
    function that outputs a table of optimal entry and exit conditions based on 
    a user-specifed asset pair and evaluation metric 
    '''
    
    max_value = perf_df.max().max()

    location = perf_df.isin([max_value])

    # Get the row index and column name
    exit_z, entry_z = location.stack().idxmax()
    
    
    output_df = pd.DataFrame({'Metric': metric, 'Max Value':max_value,'Entry Value':entry_z, 'Exit Value': exit_z,'Rolling Window':window}, index = [1]).T
    output_df = output_df.reset_index()
    output_df.columns = ['Param','Value']
    return output_df


def optimal_conditions_table_rsi(perf_df,window,metric):
    '''
    function that outputs a table of optimal entry and exit conditions based on 
    a user-specifed asset pair and evaluation metric 
    '''
    
    max_value = perf_df.max().max()

    location = perf_df.isin([max_value])

    # Get the row index and column name
    exit_z, entry_z = location.stack().idxmax()
    
    
    output_df = pd.DataFrame({'Metric': metric, 'Max Value':max_value,'Short Entry Value':entry_z, 'Long Entry Value': exit_z,'Rolling Window':window}, index = [1]).T
    output_df = output_df.reset_index()
    output_df.columns = ['Param','Value']
    return output_df

def rsi_parameter_performance(ticker1,ticker2,rolling_window, metric):
    long_entry = [0,5,10,15,20,25,30,35,40,45]
    short_entry = [55,60,65,70,75,80,85,90,95,100]
    
    heatmap_data = pd.DataFrame(index=long_entry, columns=short_entry)

    for x in long_entry:
        for y in short_entry:
            try:
                heatmap_data.loc[x,y] = calculate_metrics(simulate_rsi_trading_strategy(price_data, ticker1, ticker2, y,x,50, rolling_window)['PnL%']).set_index('Metric',drop = True).T.reset_index(drop = True)[metric][0]
            except:
                heatmap_data.loc[x,y] = 0

    return heatmap_data



#### PCA HELPER FUNCTION FOR PAGE 2 ########


def pca_analysis(index_returns,stock_returns,variance_threshold = 0.90):

    # Ensure the returns data does not have NaN values
    
    stock_returns = stock_returns.pct_change()
    index_returns = index_returns.pct_change()
    
    
    # Handle NaN values
    stock_returns_filled = stock_returns.fillna(stock_returns.mean())
    index_returns = index_returns.fillna(index_returns.mean())


    # Standardize the returns data
    scaler = StandardScaler()
    securities_returns_scaled = scaler.fit_transform(stock_returns_filled)
    
    # Perform PCA on the securities returns
    pca = PCA()
    pca.fit(securities_returns_scaled)
    
    # Transform the securities returns to the PCA space
    pcs = pca.transform(securities_returns_scaled)

    # Regress index returns on the principal components
    reg = LinearRegression()
    reg.fit(pcs, index_returns)
    
    # Get the importance of each principal component
    pcs_importance = reg.coef_
    
    # Get the explained variance of each principal component
    explained_variance = pca.explained_variance_ratio_
    
    # Create a DataFrame with the PCA component loadings
    pca_loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)],       index=stock_returns.columns)
    
    # Calculate the variance explained by each security
    security_importance = np.dot(pca_loadings, pcs_importance)
    
    # Combine the importance with the securities
    importance_df = pd.DataFrame({'Security': stock_returns.columns, 'Importance': security_importance})
    
    
    # Sort by importance and select top securities
    top_securities = importance_df.sort_values(by='Importance', ascending=False).head(10)
    
    top_securities['Importance'] = range(1,len(top_securities.index)+1)
    
    conn = sqlite3.connect(db_name)
    
    # Query to fetch stock tickers
    tickers_query = '''
    SELECT ticker as Security,company_name as Company,sector as Sector,industry as Industry FROM Stocks
    '''
    
    # Filter out duplicate tickers when a company has different classes of shares
    tickers_df = pd.read_sql(tickers_query, conn)

    output_df = top_securities.merge(tickers_df,how = 'left', on = ['Security']).round(3)
    
    return output_df
