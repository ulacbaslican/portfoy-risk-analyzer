from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_PERIOD,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SIMULATIONS,
    DEFAULT_TRADING_DAYS,
)
from data_fetcher import fetch_benchmark
from risk_engine import (
    calculate_risk_metrics,
    compare_with_benchmark,
    prepare_returns,
    print_risk_report,
    run_monte_carlo,
)
from optimizer import efficient_frontier
from visualizer import plot_final_distribution, plot_simulation_paths
from visualizer import plot_efficient_frontier


def portfoy_analizi():
    print("--- Portfolio Risk Analysis Simulator (Robust Version) ---")
    varliklar = input("Enter assets (e.g., BTC-USD, AAPL, AMZN): ").split(",")
    varliklar = [v.strip().upper() for v in varliklar if v.strip()]
    if not varliklar:
        print("ERROR: No assets entered.")
        return

    try:
        anapara = float(input("Enter your initial capital: "))
    except ValueError:
        print("ERROR: Please enter a valid number.")
        return

    sonuc = prepare_returns(varliklar, period=DEFAULT_PERIOD)
    if sonuc is None:
        # BURAYA YAZACAKSIN:
        print("❌ ERROR: Data could not be fetched due to invalid tickers or connection issues. Analysis stopped.")
        print("❌ HATA: Geçersiz varlık veya bağlantı sorunu nedeniyle veri çekilemedi. Analiz durduruldu.")
        return

    data, getiriler, ortalama_getiri, kovaryans_matrisi = sonuc

    try:
        agirliklar, portfoy_sonuclari, varlik_sonuclari = run_monte_carlo(
            ortalama_getiri,
            kovaryans_matrisi,
            anapara,
            varliklar,
            gun_sayisi=DEFAULT_TRADING_DAYS,
            sim_sayisi=DEFAULT_SIMULATIONS,
        )
    except Exception as e:
        print(f"A mathematical error occurred: {e}")
        return

    plot_simulation_paths(portfoy_sonuclari, varlik_sonuclari, varliklar, anapara)
    plot_final_distribution(portfoy_sonuclari)

    metrics = calculate_risk_metrics(data, getiriler, kovaryans_matrisi, agirliklar)

    print("\nMean Returns:")
    for asset, mean_return in zip(varliklar, metrics["mean_returns"]):
        print(f"  {asset}: {mean_return:.4f}")

    print("\nCovariance Matrix:")
    print(metrics["covariance_matrix"])

    print("\nActual Annualized Risks (Volatility):")
    for asset, risk in zip(varliklar, metrics["actual_risks"]):
        print(f"  {asset}: {risk:.4f}")

    print(f"\n✅ --- ACCURATE RISK ANALYSIS (Annualized Volatility) ---")
    for asset, risk in zip(metrics["assets"], metrics["annualized_risks"]):
        print(f"  {asset}: {risk:.4f} (%{risk * 100:.2f})")

    print(f"\n🛡️  Total Portfolio Risk: {metrics['portfolio_risk']:.4f} (%{metrics['portfolio_risk'] * 100:.2f})")
    ortalama_portfoy_yolu = portfoy_sonuclari.mean(axis=1)
    son_fiyatlar = portfoy_sonuclari[-1, :]
    print("\n" + "=" * 50)
    print(" PORTFOLIO PROBABILITY AND RISK COMPARISON REPORT ")
    print_risk_report(
        returns=getiriler,
        weights=agirliklar,
        portfolio_matrix=portfoy_sonuclari,
        final_values=son_fiyatlar,
        initial_capital=anapara,
        risk_free_rate=DEFAULT_RISK_FREE_RATE,
        confidence=DEFAULT_CONFIDENCE,
    )
    print("=" * 50)

    benchmark_ticker = input("Enter benchmark ticker (^GSPC / ^XU100 / press Enter to skip): ").strip().upper()

    if benchmark_ticker:
        benchmark_returns = fetch_benchmark(benchmark_ticker=benchmark_ticker, period=DEFAULT_PERIOD)
        if benchmark_returns is None or benchmark_returns.empty:
            print("Benchmark comparison skipped because benchmark data could not be fetched.")
            return

        if hasattr(benchmark_returns, "ndim") and benchmark_returns.ndim > 1:
            benchmark_returns = benchmark_returns.squeeze()

        comparison = compare_with_benchmark(getiriler, benchmark_returns, agirliklar)
        portfolio_daily_returns = getiriler.dot(agirliklar)
        aligned_portfolio_returns, aligned_benchmark_returns = portfolio_daily_returns.align(benchmark_returns, join="inner")

        portfolio_annual_return = float(portfolio_daily_returns.mean() * DEFAULT_TRADING_DAYS)
        benchmark_annual_return = float(aligned_benchmark_returns.mean() * DEFAULT_TRADING_DAYS)
        portfolio_annual_volatility = float(aligned_portfolio_returns.std(ddof=1) * (DEFAULT_TRADING_DAYS ** 0.5))
        benchmark_annual_volatility = float(aligned_benchmark_returns.std(ddof=1) * (DEFAULT_TRADING_DAYS ** 0.5))

        print("\n" + "=" * 72)
        print(" PORTFOLIO VS BENCHMARK COMPARISON ")
        print("=" * 72)
        print(f"{'Metric':<24}{'Portfolio':>22}{'Benchmark':>22}")
        print("-" * 72)
        print(f"{'Annualized Return':<24}{portfolio_annual_return:>22.4f}{benchmark_annual_return:>22.4f}")
        print(f"{'Annualized Volatility':<24}{portfolio_annual_volatility:>22.4f}{benchmark_annual_volatility:>22.4f}")
        print(f"{'Sharpe Ratio':<24}{comparison['portfolio_sharpe']:>22.4f}{comparison['benchmark_sharpe']:>22.4f}")
        print(f"{'Beta':<24}{comparison['beta']:>22.4f}{'N/A':>22}")
        print("=" * 72)

    if len(varliklar) < 2:
        print("Efficient Frontier skipped: requires at least 2 assets.")
    else:
        frontier_risks, frontier_returns = efficient_frontier(getiriler)
        plot_efficient_frontier(
            frontier_risks,
            frontier_returns,
            metrics["portfolio_risk"],
            float(getiriler.dot(agirliklar).mean() * DEFAULT_TRADING_DAYS),
        )

if __name__ == "__main__":
    portfoy_analizi()
