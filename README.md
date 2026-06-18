# Portfolio Risk Analysis Tool

## Project Description

This project is a Python-based portfolio risk analysis tool that downloads market data, simulates portfolio outcomes with Monte Carlo methods, and reports key risk statistics for a selected set of assets. It is designed for quick scenario analysis of multi-asset portfolios using historical price data and probabilistic forecasting.

## Features

- Monte Carlo simulation of portfolio value paths and asset-level paths
- Sharpe ratio calculation for risk-adjusted performance assessment
- Maximum drawdown measurement to quantify peak-to-trough loss
- CVaR calculation to estimate tail risk at a chosen confidence level
- Win probability estimation to show the chance of finishing above initial capital

## Installation

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the application from the project root with:

```bash
python main.py
```

When prompted, enter a comma-separated list of tickers such as `AAPL,MSFT,BTC-USD`, then provide the initial capital amount.

## Project Structure

- `data_fetcher.py` - Downloads historical price data for the requested assets.
- `risk_engine.py` - Calculates returns, runs Monte Carlo simulations, and computes risk metrics.
- `visualizer.py` - Generates charts for simulated portfolio paths and final outcome distributions.
- `main.py` - Coordinates user input, analysis workflow, and reporting.
