"""
02 - Multiple regression

Part A: omitted variable bias - what happens to the return to education
        when ability (IQ) is left out of the wage equation.
Part B: the Frisch-Waugh-Lovell (FWL) theorem - a coefficient in a multiple
        regression equals the slope from regressing y on the "partialled-out"
        version of that regressor.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import wooldridge as woo

from helpers import coef_table, section


def omitted_variable_bias():
    section("Part A: omitted variable bias (wage, education and IQ)")
    wage2 = woo.dataWoo("wage2")

    short = smf.ols("np.log(wage) ~ educ", data=wage2).fit()       # IQ omitted
    long = smf.ols("np.log(wage) ~ educ + IQ", data=wage2).fit()   # IQ included
    aux = smf.ols("IQ ~ educ", data=wage2).fit()                   # IQ on educ

    print("Short regression: log(wage) ~ educ")
    print(coef_table(short))
    print("\nLong regression: log(wage) ~ educ + IQ")
    print(coef_table(long))
    print("\nAuxiliary regression: IQ ~ educ")
    print(coef_table(aux))

    # Identity: short slope = long slope + (effect of IQ) * (IQ ~ educ slope)
    b1_short = short.params["educ"]
    b1_implied = long.params["educ"] + long.params["IQ"] * aux.params["educ"]

    b0_short = short.params["Intercept"]
    b0_implied = long.params["Intercept"] + long.params["IQ"] * aux.params["Intercept"]

    check = pd.DataFrame({
        "short regression": [b1_short, b0_short],
        "long + IQ effect x aux": [b1_implied, b0_implied],
    }, index=["educ slope", "intercept"])
    print("\nVerifying the omitted variable identity:")
    print(check.round(6))

    bias = b1_short - long.params["educ"]
    print(f"\nLeaving IQ out raises the estimated return to a year of education "
          f"from {long.params['educ']:.4f} to {b1_short:.4f} "
          f"(upward bias of {bias:.4f}),\nbecause IQ raises wages and is "
          f"positively related to education.")


def frisch_waugh_lovell():
    section("Part B: Frisch-Waugh-Lovell theorem (wage1)")
    wage1 = woo.dataWoo("wage1").copy()

    full = smf.ols("np.log(wage) ~ educ + exper + tenure", data=wage1).fit()
    print("Full regression: log(wage) ~ educ + exper + tenure")
    print(coef_table(full))

    # Slope: remove the part of educ explained by exper and tenure,
    # then regress log(wage) on what is left.
    wage1["educ_resid"] = smf.ols("educ ~ exper + tenure", data=wage1).fit().resid
    fwl_slope = smf.ols("np.log(wage) ~ educ_resid", data=wage1).fit()

    # Intercept: the constant is just another regressor (a column of ones).
    # Partial the other regressors out of it (no intercept in this step),
    # then regress log(wage) on the residual, again without an intercept.
    wage1["const"] = 1.0
    wage1["const_resid"] = smf.ols("const ~ educ + exper + tenure - 1",
                                   data=wage1).fit().resid
    fwl_const = smf.ols("np.log(wage) ~ const_resid - 1", data=wage1).fit()

    check = pd.DataFrame({
        "full regression": [full.params["educ"], full.params["Intercept"]],
        "FWL two-step": [fwl_slope.params["educ_resid"],
                         fwl_const.params["const_resid"]],
    }, index=["educ slope", "intercept"])
    print("\nFull regression vs. two-step FWL estimates:")
    print(check.round(6))


if __name__ == "__main__":
    omitted_variable_bias()
    frisch_waugh_lovell()
