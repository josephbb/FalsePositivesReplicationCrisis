import arviz as az
import polars as pl
import numpy as np


#from https://stackoverflow.com/questions/75555521/polars-equivalent-of-pandas-factorize
def factorize(series: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    name = series.name
    df = series.to_frame()
    df_ranked = df.unique(maintain_order=True).with_row_index(name=f"{name}_index")
    uniques = df_ranked[name].to_numpy()
    codes = df.join(df_ranked, how="left", on=name)[f"{name}_index"].to_numpy()
    return codes, uniques

def check_diagnostics(idata, var_names=["~mu", "~alpha", "~B_raw"]):
    """
    Check diagnostics for an arviz InferenceData object

    Parameters:
        idata (arviz.InferenceData): The inference data object.
        var_names (list, optional): List of variable names to include in the diagnostics.

    Returns:
        dict: A dictionary containing the diagnostics.

    """
    diagnostics = {}
    diagnostics["Total divergences"] = idata.sample_stats["diverging"].sum().values
    diagnostics["Max Rhat"] = az.summary(idata, var_names=var_names)["r_hat"].max()
    diagnostics["Min effective sample size(bulk)"] = az.summary(
        idata, var_names=var_names
    )["ess_bulk"].min()
    diagnostics["Min effective sample size(tail)"] = az.summary(
        idata, var_names=var_names
    )["ess_tail"].min()
    diagnostics["Min BFMI"] = az.bfmi(idata).min()

    print("Total divergences: ", diagnostics["Total divergences"])
    print("Max Rhat: ", diagnostics["Max Rhat"])
    print(
        "Min effective sample size(bulk): ",
        diagnostics["Min effective sample size(bulk)"],
    )
    print(
        "Min effective sample size(tail): ",
        diagnostics["Min effective sample size(tail)"],
    )
    print("Min BFMI: ", diagnostics["Min BFMI"])

    # check if any of the diagnostics are bad
    if (
        diagnostics["Total divergences"] > 0
        or diagnostics["Max Rhat"] > 1.1
        or diagnostics["Min effective sample size(bulk)"] < 200
        or diagnostics["Min effective sample size(tail)"] < 200
        or diagnostics["Min BFMI"] < 0.3
    ):
        print("Bad diagnostics, increase tuning or check model")
        diagnostics["success"] = False
        return diagnostics
    else:
        print("Good diagnostics, continue")
        diagnostics["success"] = True
        return diagnostics
