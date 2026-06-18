import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


def _prepare_returns_matrix(returns_df):
	if not isinstance(returns_df, pd.DataFrame):
		returns_df = pd.DataFrame(returns_df)

	clean_returns = returns_df.dropna().copy()
	if clean_returns.empty:
		raise ValueError("returns_df must contain at least one row of valid returns")

	return clean_returns


def _annualized_metrics(returns_df):
	mean_returns = returns_df.mean() * TRADING_DAYS
	covariance_matrix = returns_df.cov() * TRADING_DAYS
	return mean_returns, covariance_matrix


def _portfolio_return(weights, mean_returns):
	return float(np.dot(weights, mean_returns))


def _portfolio_variance(weights, covariance_matrix):
	return float(np.dot(weights.T, np.dot(covariance_matrix, weights)))


def _portfolio_volatility(weights, covariance_matrix):
	return float(np.sqrt(max(_portfolio_variance(weights, covariance_matrix), 0.0)))


def efficient_frontier(returns_df, n_points=50):
	"""Build an efficient frontier by minimizing volatility for a range of target returns."""
	clean_returns = _prepare_returns_matrix(returns_df)
	mean_returns, covariance_matrix = _annualized_metrics(clean_returns)

	min_target = float(mean_returns.min())
	max_target = float(mean_returns.max())
	target_returns = np.linspace(min_target, max_target, n_points)

	frontier_risks = []
	frontier_returns = []
	num_assets = clean_returns.shape[1]
	bounds = tuple((0.0, 1.0) for _ in range(num_assets))
	initial_weights = np.repeat(1.0 / num_assets, num_assets)

	for target_return in target_returns:
		constraints = (
			{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
			{"type": "eq", "fun": lambda weights, target=target_return: _portfolio_return(weights, mean_returns.values) - target},
		)

		result = minimize(
			_portfolio_volatility,
			initial_weights,
			args=(covariance_matrix.values,),
			method="SLSQP",
			bounds=bounds,
			constraints=constraints,
		)

		if result.success:
			optimal_weights = result.x
			frontier_risks.append(_portfolio_volatility(optimal_weights, covariance_matrix.values))
			frontier_returns.append(_portfolio_return(optimal_weights, mean_returns.values))

	return frontier_risks, frontier_returns


def find_max_sharpe(returns_df, risk_free_rate=0.02):
	"""Return portfolio weights that maximize the annualized Sharpe ratio."""
	clean_returns = _prepare_returns_matrix(returns_df)
	mean_returns, covariance_matrix = _annualized_metrics(clean_returns)

	num_assets = clean_returns.shape[1]
	bounds = tuple((0.0, 1.0) for _ in range(num_assets))
	initial_weights = np.repeat(1.0 / num_assets, num_assets)

	def negative_sharpe(weights):
		portfolio_return = _portfolio_return(weights, mean_returns.values)
		portfolio_volatility = _portfolio_volatility(weights, covariance_matrix.values)
		if portfolio_volatility == 0:
			return np.inf
		return -((portfolio_return - risk_free_rate) / portfolio_volatility)

	constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)

	result = minimize(
		negative_sharpe,
		initial_weights,
		method="SLSQP",
		bounds=bounds,
		constraints=constraints,
	)

	if not result.success:
		return initial_weights

	return result.x
