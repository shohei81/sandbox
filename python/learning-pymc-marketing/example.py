import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings

    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
    from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
    from pymc_extras.prior import Prior

    warnings.filterwarnings("ignore", category=FutureWarning)

    az.style.use("arviz-darkgrid")
    plt.rcParams["figure.figsize"] = [12, 7]
    plt.rcParams["figure.dpi"] = 100

    # %load_ext autoreload
    # %autoreload 2
    # %config InlineBackend.figure_format = "retina"
    return geometric_adstock, logistic_saturation, np, pd, plt, sns


@app.cell
def _(np, pd):
    seed: int = sum(map(ord, "mmm"))
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    # date range
    min_date = pd.to_datetime("2018-04-01")
    max_date = pd.to_datetime("2021-09-01")

    df = pd.DataFrame(
        data={"date_week": pd.date_range(start=min_date, end=max_date, freq="W-MON")}
    ).assign(
        year=lambda x: x["date_week"].dt.year,
        month=lambda x: x["date_week"].dt.month,
        dayofyear=lambda x: x["date_week"].dt.dayofyear,
    )

    n = df.shape[0]
    print(f"Number of observations: {n}")
    return df, n, rng


@app.cell
def _(df, n, np, plt, rng: "np.random.Generator", sns):
    # media data
    x1 = rng.uniform(low=0.0, high=1.0, size=n)
    df["x1"] = np.where(x1 > 0.9, x1, x1 / 2)

    x2 = rng.uniform(low=0.0, high=1.0, size=n)
    df["x2"] = np.where(x2 > 0.8, x2, 0)


    fig, ax = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 7), sharex=True, sharey=True, layout="constrained"
    )
    sns.lineplot(x="date_week", y="x1", data=df, color="C0", ax=ax[0])
    sns.lineplot(x="date_week", y="x2", data=df, color="C1", ax=ax[1])
    ax[1].set(xlabel="date")
    fig.suptitle("Media Costs Data", fontsize=16)
    return


@app.cell
def _(df, geometric_adstock):
    # apply geometric adstock transformation
    alpha1: float = 0.4
    alpha2: float = 0.2

    df["x1_adstock"] = (
        geometric_adstock(x=df["x1"].to_numpy(), alpha=alpha1, l_max=8, normalize=True)
        .eval()
        .flatten()
    )

    df["x2_adstock"] = (
        geometric_adstock(x=df["x2"].to_numpy(), alpha=alpha2, l_max=8, normalize=True)
        .eval()
        .flatten()
    )
    return


@app.cell
def _(df, logistic_saturation):
    # apply saturation transformation
    lam1: float = 4.0
    lam2: float = 3.0

    df["x1_adstock_saturated"] = logistic_saturation(
        x=df["x1_adstock"].to_numpy(), lam=lam1
    ).eval()

    df["x2_adstock_saturated"] = logistic_saturation(
        x=df["x2_adstock"].to_numpy(), lam=lam2
    ).eval()
    return


@app.cell
def _(df, plt, sns):
    fig_media, ax_media = plt.subplots(
        nrows=3, ncols=2, figsize=(16, 9),
        sharex=True, sharey=False, layout="constrained"
    )

    sns.lineplot(x="date_week", y="x1", data=df, color="C0", ax=ax_media[0, 0])
    sns.lineplot(x="date_week", y="x2", data=df, color="C1", ax=ax_media[0, 1])

    sns.lineplot(x="date_week", y="x1_adstock", data=df, color="C0", ax=ax_media[1, 0])
    sns.lineplot(x="date_week", y="x2_adstock", data=df, color="C1", ax=ax_media[1, 1])

    sns.lineplot(
        x="date_week", y="x1_adstock_saturated",
        data=df, color="C0", ax=ax_media[2, 0]
    )
    sns.lineplot(
        x="date_week", y="x2_adstock_saturated",
        data=df, color="C1", ax=ax_media[2, 1]
    )

    fig_media.suptitle("Media Costs Data - Transformed", fontsize=16)
    fig_media

    return


@app.cell
def _(df, plt, sns):
    fig_ts, ax_ts = plt.subplots()

    sns.lineplot(
        x="date_week", y="trend",
        color="C2", label="trend", data=df, ax=ax_ts
    )
    sns.lineplot(
        x="date_week", y="seasonality",
        color="C3", label="seasonality", data=df, ax=ax_ts
    )

    ax_ts.legend(loc="upper left")
    ax_ts.set(
        title="Trend & Seasonality Components",
        xlabel="date",
        ylabel=None
    )

    fig_ts

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
