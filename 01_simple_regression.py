"""
01 - Simple linear regression

Part A: CEO salary and tenure (log-level model on real data).
Part B: Monte Carlo check of the algebraic properties of OLS residuals.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import wooldridge as woo

from helpers import FIG_DIR, coef_table, section


def ceo_salary_and_tenure():
    section("Part A: CEO salary vs. tenure")
    ceo = woo.dataWoo("ceosal2")

    print(f"Average salary (thousand USD): {ceo['salary'].mean():.2f}")
    print(f"Average tenure (years):        {ceo['ceoten'].mean():.2f}")
    print(f"CEOs in their first year:      {(ceo['ceoten'] == 0).sum()}")
    print(f"Longest tenure (years):        {ceo['ceoten'].max()}")

    # Log-level model: the slope is the approximate % change in salary
    # for one more year as CEO.
    results = smf.ols("np.log(salary) ~ ceoten", data=ceo).fit()
    print("\nlog(salary) ~ ceoten")
    print(coef_table(results))

    slope = results.params["ceoten"]
    print(f"\nOne extra year as CEO is associated with about "
          f"{slope * 100:.2f}% higher salary (R² = {results.rsquared:.3f}).")


def simulate_once(rng, n=500, beta0=1.0, beta1=2.0, error_sd=6.0):
    """Simulate y = beta0 + beta1 * x + u and fit OLS."""
    x = rng.uniform(0, 10, size=n)
    u = rng.normal(0, error_sd, size=n)   # variance = error_sd ** 2 = 36
    y = beta0 + beta1 * x + u
    data = pd.DataFrame({"y": y, "x": x})
    results = smf.ols("y ~ x", data=data).fit()
    return data, u, results


def residual_properties():
    section("Part B: properties of OLS residuals (simulation)")
    rng = np.random.default_rng(seed=2024)

    rows = []
    for run in range(1, 3):
        data, u, results = simulate_once(rng)
        u_hat = results.resid
        rows.append({
            "run": run,
            "b0_hat": results.params["Intercept"],
            "b1_hat": results.params["x"],
            "sum(u_hat)": u_hat.sum(),
            "sum(x * u_hat)": (data["x"] * u_hat).sum(),
            "sum(u)": u.sum(),
            "sum(x * u)": (data["x"] * u).sum(),
        })

    summary = pd.DataFrame(rows).set_index("run")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary)

    print("\nThe residuals u_hat sum to zero and are uncorrelated with x by "
          "construction\n(OLS first-order conditions). The true errors u do not "
          "satisfy this exactly,\nand the estimates change from run to run "
          "because each sample is different.")

    # Plot the last simulated sample: fitted line vs. true line
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(data["x"], data["y"], s=10, alpha=0.5, label="Simulated data")
    grid = np.linspace(0, 10, 100)
    ax.plot(grid, 1 + 2 * grid, color="black", linewidth=2, label="True line")
    ax.plot(grid, results.params["Intercept"] + results.params["x"] * grid,
            color="red", linestyle="--", linewidth=2, label="OLS fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated data: y = 1 + 2x + u,  u ~ N(0, 36)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_simulated_regression.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ceo_salary_and_tenure()
    residual_properties()
