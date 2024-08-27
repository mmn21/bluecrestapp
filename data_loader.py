#!/usr/bin/env python
# coding: utf-8




import yfinance as yf
import pandas as pd
import sqlite3
import csv
from datetime import datetime




def get_tickers():
    """
    Collects the tickers for the components of the SP500, Nasdaq100, and Russell 2000 indices as well as the 
    indices themselves
    """

    sp500 = sorted(pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist())
    nasdaq = sorted(pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#Components')[4]['Ticker'].tolist())
    russel = sorted(pd.read_csv('russell_components.csv', on_bad_lines = 'skip')['Ticker'].tolist())
    all_tickers = sp500 + nasdaq + russel + ['^SPX','NDX','^RUT']
    all_tickers = sorted(list(dict.fromkeys(all_tickers)))
    
    return all_tickers

# Function to create the database and tables
def create_database(db_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create the Stocks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Stocks (
            stock_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            company_name TEXT,
            sector TEXT,
            industry TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create the DailyPrices table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS DailyPrices (
            price_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER,
            price_date DATE NOT NULL,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            adjusted_close_price REAL,
            volume INTEGER,
            FOREIGN KEY (stock_id) REFERENCES Stocks(stock_id),
            UNIQUE(stock_id, price_date)
        )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Function to insert stock metadata into the Stocks table
def insert_stock_metadata(conn, ticker):
    # Fetch stock metadata using yfinance
    stock = yf.Ticker(ticker)
    info = stock.info

    # Insert metadata into the Stocks table
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO Stocks (ticker, company_name, sector, industry)
        VALUES (?, ?, ?, ?)
    ''', (ticker, info.get('longName'), info.get('sector'), info.get('industry')))
    
    conn.commit()
    return cursor.lastrowid

# Function to insert end-of-day prices into the DailyPrices table
def insert_daily_prices(conn, stock_id, ticker):
    # Fetch historical price data using yfinance
    data = yf.download(ticker, start = '2019-01-01', end = datetime.today().strftime('%Y-%m-%d'), interval='1d')
    
    # Prepare data for insertion
    data['stock_id'] = stock_id
    data = data.reset_index()
    data = data.rename(columns={
        'Date': 'price_date',
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Close': 'close_price',
        'Adj Close': 'adjusted_close_price',
        'Volume': 'volume'
    })

    # Insert data into the DailyPrices table
    data.to_sql('DailyPrices', conn, if_exists='append', index=False)

# Main function to create database and insert data
def main():
    db_name = 'stocks_data.db'
    create_database(db_name)
    
    # List of tickers to insert into the database
    tickers = get_tickers()  

    # Connect to the database
    conn = sqlite3.connect(db_name)

    # Insert stock metadata and daily prices for each ticker
    for ticker in tickers:
        stock_id = insert_stock_metadata(conn, ticker)
        insert_daily_prices(conn, stock_id, ticker)
    
    # Close the connection
    conn.close()

if __name__ == '__main__':
    main()

