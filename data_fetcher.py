import requests
import pandas as pd
import yfinance as yf


def fetch_price_data(assets, period="2y"):
    """Download adjusted close prices for the selected assets."""
    print(f"\nDownloading data for {assets}...")
    try:
        data = yf.download(assets, period=period, auto_adjust=True)
        if data.empty:
            print(f"Hatalı ticker / Invalid ticker: {assets}. Veri boş geldi / downloaded data is empty.")
            return None
        return data["Close"]
    except (ConnectionError, requests.exceptions.ConnectionError):
        print("İnternet bağlantısı yok / Connection error: Veri indirilemedi / could not download data.")
        return None
    except Exception as exc:
        print(f"Beklenmeyen hata / Unexpected error: {exc}")
        return None


def fetch_benchmark(benchmark_ticker="^GSPC", period="2y"):
    # ^GSPC is the S&P 500 index, and ^XU100 is the BIST 100 index.
    """Download benchmark closing prices and return daily benchmark returns."""
    print(f"\nDownloading benchmark data for {benchmark_ticker}...")
    try:
        data = yf.download(benchmark_ticker, period=period, auto_adjust=True)
        if data.empty:
            print(f"Hatalı benchmark ticker / Invalid benchmark ticker: {benchmark_ticker}. Veri boş geldi / downloaded data is empty.")
            return None

        close_prices = data["Close"]
        if getattr(close_prices, "ndim", 1) > 1:
            close_prices = close_prices.squeeze()
            if getattr(close_prices, "ndim", 1) > 1:
                close_prices = close_prices.iloc[:, 0]
        if hasattr(close_prices, "squeeze"):
            close_prices = close_prices.squeeze()
        if hasattr(close_prices, "iloc") and close_prices.ndim > 1:
            close_prices = close_prices.iloc[:, 0]
        close_prices = pd.Series(close_prices.values, index=close_prices.index)
        daily_returns = close_prices.pct_change().dropna()
        daily_returns.name = f"{benchmark_ticker}_returns"
        return daily_returns
    except (ConnectionError, requests.exceptions.ConnectionError):
        print("İnternet bağlantısı yok / Connection error: Benchmark verisi indirilemedi / could not download benchmark data.")
        return None
    except Exception as exc:
        print(f"Beklenmeyen hata / Unexpected error: {exc}")
        return None

