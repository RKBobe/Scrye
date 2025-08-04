import os
import pandas as pd
import requests
import yfinance as yf

def get_top_coins(limit=20):
    """Fetches the top cryptocurrencies by market cap from the CoinGecko API."""
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': limit,
        'page': 1,
        'sparkline': False
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return [coin['symbol'].upper() + '-USD' for coin in response.json()]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching top coins from CoinGecko: {e}")
        return None

def download_coin_data(ticker_symbol, data_dir='data', period='1y', interval='1d'):
    """Downloads historical data for a given ticker symbol and saves it to a CSV file."""
    try:
        data = yf.download(ticker_symbol, period=period, interval=interval)
        if data.empty:
            print(f"No data found for {ticker_symbol}, skipping.")
            return

        file_name = f'{ticker_symbol}_data.csv'
        file_path = os.path.join(data_dir, file_name)
        data.to_csv(file_path)
        print(f"Data for {ticker_symbol} saved to {file_path}")

    except Exception as e:
        print(f"Could not download or save data for {ticker_symbol}: {e}")

def main():
    """Main function to get top coins and download their data."""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    print("Fetching the top 20 cryptocurrencies by market cap...")
    top_tickers = get_top_coins(20)

    if top_tickers:
        print("Starting data download for the top 20 coins...")
        for ticker in top_tickers:
            download_coin_data(ticker, data_dir)
        print("\nData download complete! 🎉")

if __name__ == "__main__":
    main()