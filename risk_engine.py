import numpy as np

from data_fetcher import fetch_price_data


def prepare_returns(assets, period="2y"):
    data = fetch_price_data(assets, period=period)
    if data is None or data.empty:
        return None
    getiriler = data.pct_change().dropna()

    if getiriler.empty:
        print("ERROR: No common data found for the selected assets. Please check the assets!")
        return None

    ortalama_getiri = getiriler.mean()
    kovaryans_matrisi = getiriler.cov()

    if np.any(np.isnan(kovaryans_matrisi)):
        print("ERROR: There are missing values in the covariance matrix. Insufficient data set.")
        return None

    return data, getiriler, ortalama_getiri, kovaryans_matrisi

def run_monte_carlo(ortalama_getiri, kovaryans_matrisi, anapara, varliklar, gun_sayisi=252, sim_sayisi=1000):
    agirliklar = np.array([1 / len(varliklar)] * len(varliklar))
    portfoy_sonuclari = np.zeros((gun_sayisi, sim_sayisi))
    varlik_sonuclari = np.zeros((gun_sayisi, sim_sayisi, len(varliklar)))

    for i in range(sim_sayisi):
        rastgele_getiriler = np.random.multivariate_normal(ortalama_getiri, kovaryans_matrisi, gun_sayisi)
        gunluk_portfoy_getirisi = np.dot(rastgele_getiriler, agirliklar)
        portfoy_sonuclari[:, i] = anapara * (1 + gunluk_portfoy_getirisi).cumprod()
        varlik_sonuclari[:, i, :] = (anapara * agirliklar) * (1 + rastgele_getiriler).cumprod(axis=0)

    return agirliklar, portfoy_sonuclari, varlik_sonuclari

def calculate_risk_metrics(data, getiriler, kovaryans_matrisi, agirliklar, gun_sayisi=252):
    actual_risks = np.sqrt(np.diag(kovaryans_matrisi)) * np.sqrt(gun_sayisi)
    annualized_risks = getiriler.std() * np.sqrt(gun_sayisi)
    portfolio_risk = np.sqrt(np.dot(agirliklar.T, np.dot(kovaryans_matrisi, agirliklar))) * np.sqrt(gun_sayisi)
    portfolio_return = float(np.dot(getiriler.mean(), agirliklar) * gun_sayisi)

    return {
        "mean_returns": getiriler.mean(),
        "covariance_matrix": kovaryans_matrisi,
        "actual_risks": actual_risks,
        "annualized_risks": annualized_risks,
        "portfolio_risk": portfolio_risk,
        "portfolio_return": portfolio_return,
        "assets": data.columns,
    }

def sharpe_ratio(returns, weights, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio for a portfolio.

    returns: daily returns as pandas Series (single asset) or DataFrame (assets x days)
    weights: array-like weights for assets (length must match columns if DataFrame)
    risk_free_rate: annual risk-free rate (float)

    Returns a float (annualized Sharpe ratio).
    """
    # Convert returns to numpy daily portfolio returns
    if hasattr(returns, "dot") and getattr(returns, "shape", None) and len(getattr(returns, "shape")) == 2:
        # DataFrame-like: columns should match weights
        port_daily = np.asarray(returns.dot(weights))
    else:
        # Series-like or 1D array
        port_daily = np.asarray(returns)
        # if multiple weights were provided for single series, take weighted sum
        if np.ndim(weights) > 0 and len(weights) == 1:
            pass

    # Annualize
    mean_annual = np.nanmean(port_daily) * 252
    vol_annual = np.nanstd(port_daily, ddof=1) * np.sqrt(252)
    excess = mean_annual - risk_free_rate
    if vol_annual == 0 or np.isnan(vol_annual):
        return 0.0
    return float(excess / vol_annual)


def compare_with_benchmark(portfolio_returns_series, benchmark_returns_series, weights):
    """Compare a portfolio against a benchmark using annualized risk and return metrics."""
    if hasattr(portfolio_returns_series, "dot") and getattr(portfolio_returns_series, "shape", None) and len(getattr(portfolio_returns_series, "shape")) == 2:
        portfolio_daily_returns = np.asarray(portfolio_returns_series.dot(weights))
    else:
        portfolio_daily_returns = np.asarray(portfolio_returns_series)

    if hasattr(portfolio_returns_series, "align") and hasattr(benchmark_returns_series, "align"):
        portfolio_daily = portfolio_returns_series.dot(weights) if hasattr(portfolio_returns_series, "dot") and portfolio_returns_series.ndim == 2 else portfolio_returns_series.squeeze()
        aligned_portfolio, aligned_benchmark = portfolio_daily.align(benchmark_returns_series, join="inner", axis=0)
        portfolio_daily_returns = np.asarray(aligned_portfolio)
        benchmark_daily_returns = np.asarray(aligned_benchmark)
    else:
        benchmark_daily_returns = np.asarray(benchmark_returns_series)

    portfolio_return = float(np.nanmean(portfolio_daily_returns) * 252)
    benchmark_return = float(np.nanmean(benchmark_daily_returns) * 252)

    portfolio_volatility = float(np.nanstd(portfolio_daily_returns, ddof=1) * np.sqrt(252))
    benchmark_volatility = float(np.nanstd(benchmark_daily_returns, ddof=1) * np.sqrt(252))

    portfolio_sharpe = float((portfolio_return - 0.02) / portfolio_volatility) if np.isfinite(portfolio_volatility) and portfolio_volatility != 0 else 0.0
    benchmark_sharpe = float((benchmark_return - 0.02) / benchmark_volatility) if np.isfinite(benchmark_volatility) and benchmark_volatility != 0 else 0.0

    min_length = min(len(portfolio_daily_returns), len(benchmark_daily_returns))
    aligned_portfolio = portfolio_daily_returns[:min_length]
    aligned_benchmark = benchmark_daily_returns[:min_length]

    covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
    benchmark_variance = np.var(aligned_benchmark)
    beta = float(covariance / benchmark_variance) if np.isfinite(benchmark_variance) and benchmark_variance != 0 else 0.0

    return {
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "portfolio_sharpe": portfolio_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "beta": beta,
    }


def max_drawdown(portfolio_values):
    """Return the maximum drawdown (most negative peak-to-trough) as a negative float."""
    pv = np.asarray(portfolio_values)
    if pv.size == 0:
        return 0.0
    cumulative_max = np.maximum.accumulate(pv)
    drawdown = (pv - cumulative_max) / cumulative_max
    return float(np.min(drawdown))


def var(final_values, confidence=0.95):
    """Return the Value at Risk threshold at the requested confidence level."""
    vals = np.asarray(final_values)
    if vals.size == 0:
        return 0.0
    var_pct = (1.0 - confidence) * 100.0
    return float(np.percentile(vals, var_pct))


def cvar(final_values, confidence=0.95):
    """Conditional Value at Risk: average of values below the VaR threshold."""
    vals = np.asarray(final_values)
    if vals.size == 0:
        return 0.0
    var_threshold = var(final_values, confidence=confidence)
    tail = vals[vals <= var_threshold]
    if tail.size == 0:
        return float(var_threshold)
    return float(np.mean(tail))


def win_probability(final_values, initial_capital):
    """Return percentage of simulations where final value > initial capital."""
    vals = np.asarray(final_values)
    if vals.size == 0:
        return 0.0
    prob = np.mean(vals > initial_capital)
    return float(prob * 100.0)


def print_risk_report(returns, weights, portfolio_matrix, final_values, initial_capital, risk_free_rate=0.02, confidence=0.95):
    """Compute and print risk metrics clearly labeled."""
    sr = sharpe_ratio(returns, weights, risk_free_rate=risk_free_rate)
    portfolio_matrix = np.asarray(portfolio_matrix)
    if portfolio_matrix.size == 0:
        mdd = 0.0
    elif portfolio_matrix.ndim == 1:
        mdd = max_drawdown(portfolio_matrix)
    else:
        drawdowns = [max_drawdown(portfolio_matrix[:, idx]) for idx in range(portfolio_matrix.shape[1])]
        mdd = float(np.mean(drawdowns)) if drawdowns else 0.0
    vr = var(final_values, confidence=confidence)
    cv = cvar(final_values, confidence=confidence)
    wp = win_probability(final_values, initial_capital)

    print("Risk Report:")
    print(f"  Sharpe Ratio (annualized): {sr:.4f}")
    print(f"  Max Drawdown: {mdd:.4f}")
    print(f"  VaR ({confidence:.0%}): {vr:.4f}")
    print(f"  CVaR (confidence={confidence:.2f}): {cv:.4f}")
    print(f"  Win Probability (final > initial {initial_capital}): {wp:.2f}%")




