"""
03 - Statistical inference in multiple regression

Part A: Monte Carlo study of confidence interval coverage.
Part B: testing a linear restriction (campaign spending and vote share),
        three equivalent ways: F test, reparameterised t test, manual t test.
Part C: confidence interval for a linear combination of coefficients
        (house prices).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
import wooldridge as woo

from helpers import FIG_DIR, coef_table, section


# ----------------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------------
def ci_coverage(n_reps, n=300, betas=(0.4, -1.46, 2.5),
                levels=(0.95, 0.99), seed=2024):
    """
    Share of simulated samples whose confidence interval contains the true
    parameter. x1 and x2 are held fixed across replications; only the
    error term is redrawn.
    """
    rng = np.random.default_rng(seed)
    b0, b1, b2 = betas
    x1 = rng.uniform(0, 1, size=n)
    x2 = rng.normal(0.78, 1, size=n)
    true = pd.Series({"Intercept": b0, "x1": b1, "x2": b2})

    hits = {level: [] for level in levels}
    for _ in range(n_reps):
        y = b0 + b1 * x1 + b2 * x2 + rng.normal(size=n)
        results = smf.ols("y ~ x1 + x2",
                          data=pd.DataFrame({"y": y, "x1": x1, "x2": x2})).fit()
        for level in levels:
            ci = results.conf_int(alpha=1 - level)
            inside = (ci[0] <= true) & (true <= ci[1])
            hits[level].append(inside)

    rows = {f"{int(level * 100)}% CI": pd.DataFrame(hits[level]).mean()
            for level in levels}
    return pd.DataFrame(rows).T


def coverage_simulation():
    section("Part A: confidence interval coverage (Monte Carlo)")
    rep_counts = [50, 200, 1000]
    tables = {}
    for reps in rep_counts:
        tables[reps] = ci_coverage(reps)
        print(f"\nReplications = {reps}")
        print(tables[reps].round(3))

    print("\nWith more replications the observed coverage settles close to the "
          "nominal 95% and 99%.")

    fig, ax = plt.subplots(figsize=(7, 5))
    for level, colour in [("95% CI", "tab:blue"), ("99% CI", "tab:orange")]:
        for param, marker in zip(["Intercept", "x1", "x2"], ["o", "s", "^"]):
            ax.plot(rep_counts, [tables[r].loc[level, param] for r in rep_counts],
                    marker=marker, color=colour, alpha=0.8,
                    label=f"{level}, {param}")
        ax.axhline(float(level[:2]) / 100, color=colour, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(rep_counts, labels=[str(r) for r in rep_counts])
    ax.set_xlabel("Number of replications")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Confidence interval coverage vs. number of replications")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_ci_coverage.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------------
def test_linear_restriction(results, results_reparam, value):
    """Test H0: b(lexpendA) + b(lexpendB) = value in three equivalent ways."""
    # 1) F test on the original model
    ftest = results.f_test(f"lexpendA + lexpendB = {value}")
    f_stat, f_p = np.asarray(ftest.statistic).item(), np.asarray(ftest.pvalue).item()

    # 2) t test on the reparameterised model, where the coefficient on
    #    lexpendA *is* b1 + b2
    ttest = results_reparam.t_test(f"lexpendA = {value}")
    t_stat, t_p = np.asarray(ttest.statistic).item(), np.asarray(ttest.pvalue).item()

    # 3) The same t test by hand
    theta = results_reparam.params["lexpendA"]
    se = results_reparam.bse["lexpendA"]
    t_manual = (theta - value) / se
    p_manual = 2 * (1 - stats.t.cdf(abs(t_manual), df=results_reparam.df_resid))

    return pd.DataFrame({
        "statistic": [f_stat, t_stat, t_manual],
        "p_value": [f_p, t_p, p_manual],
    }, index=["F test", "t test (statsmodels)", "t test (manual)"]), t_stat ** 2, f_stat


def campaign_spending():
    section("Part B: does A's spending offset B's spending? (vote1)")
    vote1 = woo.dataWoo("vote1")
    print(vote1[["voteA", "lexpendA", "lexpendB", "prtystrA"]].describe().round(2))

    results = smf.ols("voteA ~ lexpendA + lexpendB + prtystrA", data=vote1).fit()
    print("\nvoteA ~ lexpendA + lexpendB + prtystrA")
    print(coef_table(results))

    # Reparameterise: b1*A + b2*B = (b1 + b2)*A + b2*(B - A)
    vote1 = vote1.assign(lexpendB_minus_A=vote1["lexpendB"] - vote1["lexpendA"])
    reparam = smf.ols("voteA ~ lexpendA + lexpendB_minus_A + prtystrA",
                      data=vote1).fit()

    for value in [0, 1.5]:
        table, t_sq, f_stat = test_linear_restriction(results, reparam, value)
        print(f"\nH0: b(lexpendA) + b(lexpendB) = {value}")
        print(table.round(4))
        print(f"t² = {t_sq:.4f},  F = {f_stat:.4f}  ->  identical")

    print("\nH0 = 0 (spending effects offset each other) cannot be rejected at "
          "the 5% level;\nH0 = 1.5 is clearly rejected. All three methods give "
          "the same p-value.")


# ----------------------------------------------------------------------------
# Part C
# ----------------------------------------------------------------------------
def house_price_combination():
    section("Part C: CI for a linear combination (hprice1)")
    hprice1 = woo.dataWoo("hprice1")
    print(hprice1[["price", "sqrft", "bdrms"]].describe().round(2))

    # theta = effect of adding a 150 sq ft bedroom = 150*b_sqrft + b_bdrms
    results = smf.ols("np.log(price) ~ sqrft + bdrms", data=hprice1).fit()
    theta = 150 * results.params["sqrft"] + results.params["bdrms"]

    # Reparameterise so theta appears as a single coefficient (on bdrms)
    hprice1 = hprice1.assign(sqrft_adj=hprice1["sqrft"] - 150 * hprice1["bdrms"])
    reparam = smf.ols("np.log(price) ~ sqrft_adj + bdrms", data=hprice1).fit()

    print(f"\ntheta from original model:      {theta:.4f}")
    print(f"theta from reparameterised one: {reparam.params['bdrms']:.4f}")
    print(f"Standard error of theta:        {reparam.bse['bdrms']:.4f}")
    print(f"=> about {theta * 100:.1f}% higher price for a 150 sq ft bedroom")

    for level in [0.95, 0.99]:
        low, high = reparam.conf_int(alpha=1 - level).loc["bdrms"]
        print(f"{int(level * 100)}% CI for theta: [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    coverage_simulation()
    campaign_spending()
    house_price_combination()
