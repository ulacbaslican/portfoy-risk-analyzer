import json

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required

from auth import auth_bp
from db import db
from models import Analysis, User

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_PERIOD,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SIMULATIONS,
    DEFAULT_TRADING_DAYS,
)
from data_fetcher import fetch_benchmark
from optimizer import efficient_frontier
from risk_engine import (
    calculate_risk_metrics,
    cvar,
    compare_with_benchmark,
    max_drawdown,
    prepare_returns,
    print_risk_report,
    run_monte_carlo,
    sharpe_ratio,
    var as risk_var,
    win_probability,
)


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///portfoy.db"
app.config["SECRET_KEY"] = "change-this-in-production"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


app.register_blueprint(auth_bp)


def _parse_assets(assets_value):
    if not isinstance(assets_value, str):
        return None

    assets = [asset.strip().upper() for asset in assets_value.split(",") if asset.strip()]
    return assets or None


def _portfolio_drawdown(portfolio_matrix):
    if getattr(portfolio_matrix, "size", 0) == 0:
        return 0.0

    if getattr(portfolio_matrix, "ndim", 1) == 1:
        return float(max_drawdown(portfolio_matrix))

    drawdowns = [max_drawdown(portfolio_matrix[:, idx]) for idx in range(portfolio_matrix.shape[1])]
    return float(sum(drawdowns) / len(drawdowns)) if drawdowns else 0.0


@app.route("/", methods=["GET"])
def index():
    if not current_user.is_authenticated:
        return redirect(url_for("auth.login"))

    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    try:
        payload = request.get_json(silent=True) or {}
        assets = _parse_assets(payload.get("assets"))

        capital = float(payload.get("capital"))
        if not assets:
            return jsonify({"error": "assets must be a comma-separated string"}), 400

        prepared = prepare_returns(assets, period=DEFAULT_PERIOD)
        if prepared is None:
            return jsonify({"error": "unable to prepare returns for the requested assets"}), 400

        data, returns, mean_returns, covariance_matrix = prepared

        weights, portfolio_values, _ = run_monte_carlo(
            mean_returns,
            covariance_matrix,
            capital,
            assets,
            gun_sayisi=DEFAULT_TRADING_DAYS,
            sim_sayisi=DEFAULT_SIMULATIONS,
        )
        metrics = calculate_risk_metrics(data, returns, covariance_matrix, weights)
        final_values = portfolio_values[-1, :]

        print_risk_report(
            returns=returns,
            weights=weights,
            portfolio_matrix=portfolio_values,
            final_values=final_values,
            initial_capital=capital,
            risk_free_rate=DEFAULT_RISK_FREE_RATE,
            confidence=DEFAULT_CONFIDENCE,
        )

        response = {
            "mean_returns": {str(asset): float(value) for asset, value in metrics["mean_returns"].items()},
            "annualized_risks": {str(asset): float(value) for asset, value in metrics["annualized_risks"].items()},
            "covariance_matrix": metrics["covariance_matrix"].to_dict(),
            "portfolio_risk": float(metrics["portfolio_risk"]),
            "sharpe_ratio": float(sharpe_ratio(returns, weights, risk_free_rate=DEFAULT_RISK_FREE_RATE)),
            "max_drawdown": float(_portfolio_drawdown(portfolio_values)),
            "var": float(risk_var(final_values, confidence=DEFAULT_CONFIDENCE)),
            "cvar": float(cvar(final_values, confidence=DEFAULT_CONFIDENCE)),
            "win_probability": float(win_probability(final_values, capital)),
            "final_values": [float(value) for value in final_values.tolist()],
            "final_simulation_values": [float(value) for value in final_values.tolist()],
        }

        if len(assets) >= 2:
            try:
                frontier_risks, frontier_returns = efficient_frontier(returns)
                response["frontier_risks"] = [float(value) for value in frontier_risks]
                response["frontier_returns"] = [float(value) for value in frontier_returns]
                response["current_risk"] = float(metrics["portfolio_risk"])
                response["current_return"] = float(metrics["portfolio_return"])
            except Exception:
                pass

        benchmark_ticker = payload.get("benchmark")
        if isinstance(benchmark_ticker, str):
            benchmark_ticker = benchmark_ticker.strip().upper()
        else:
            benchmark_ticker = ""

        if benchmark_ticker:
            try:
                benchmark_returns = fetch_benchmark(benchmark_ticker=benchmark_ticker, period=DEFAULT_PERIOD)
                if benchmark_returns is not None and not benchmark_returns.empty:
                    benchmark_comparison = compare_with_benchmark(returns, benchmark_returns, weights)
                    response["benchmark_comparison"] = {
                        "portfolio_return": float(benchmark_comparison["portfolio_return"]),
                        "benchmark_return": float(benchmark_comparison["benchmark_return"]),
                        "portfolio_sharpe": float(benchmark_comparison["portfolio_sharpe"]),
                        "benchmark_sharpe": float(benchmark_comparison["benchmark_sharpe"]),
                        "beta": float(benchmark_comparison["beta"]),
                    }
            except Exception:
                pass

        analysis = Analysis(
            user_id=current_user.id,
            assets=payload.get("assets", ""),
            capital=capital,
            results_json=json.dumps(response),
        )
        db.session.add(analysis)
        db.session.commit()

        return jsonify(response), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)