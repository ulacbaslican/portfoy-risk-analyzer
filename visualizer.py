import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_paths(portfoy_sonuclari, varlik_sonuclari, varliklar, anapara):
    plt.figure(figsize=(10, 6))
    plt.plot(portfoy_sonuclari, color="blue", alpha=0.03)
    ortalama_portfoy_yolu = np.mean(portfoy_sonuclari, axis=1)
    plt.plot(ortalama_portfoy_yolu, color="navy", linewidth=2.5, label="Average Portfolio Path")

    renkler = plt.cm.tab10(np.linspace(0, 1, len(varliklar)))
    for idx, varlik in enumerate(varliklar):
        ortalama_varlik_yolu = np.mean(varlik_sonuclari[:, :, idx], axis=1)
        plt.plot(
            ortalama_varlik_yolu,
            linestyle="--",
            linewidth=2,
            color=renkler[idx % len(renkler)],
            label=f"{varlik} Average Path",
        )

    plt.axhline(anapara, color="red", linestyle="--", label="Initial Capital")
    plt.title(f"Portfolio Analysis: {', '.join(varliklar)}")
    plt.legend()
    plt.show()


def plot_final_distribution(portfoy_sonuclari):
    son_fiyatlar = portfoy_sonuclari[-1, :]
    plt.figure(figsize=(10, 6))
    plt.hist(son_fiyatlar, bins=50, edgecolor="black", alpha=0.7, color="skyblue")
    plt.axvline(
        np.mean(son_fiyatlar),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Average Return: {np.mean(son_fiyatlar):.2f}",
    )
    plt.axvline(
        np.percentile(son_fiyatlar, 5),
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"95% Confidence Level (VaR): {np.percentile(son_fiyatlar, 5):.2f}",
    )
    plt.title("Monte Carlo Final Results Analysis")
    plt.xlabel("Final Portfolio Value")
    plt.ylabel("Number of Scenarios")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def plot_efficient_frontier(frontier_risks, frontier_returns, portfolio_risk, portfolio_return):
    if len(frontier_risks) < 2:
        print("Efficient Frontier requires at least 2 assets. Skipping chart.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(frontier_risks, frontier_returns, color="navy", linewidth=2.5, label="Efficient Frontier")
    plt.scatter(portfolio_risk, portfolio_return, color="red", s=80, label="Current Portfolio")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()