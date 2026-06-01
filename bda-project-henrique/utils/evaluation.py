# utils/evaluation.py

from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from pyspark.sql.window import Window

def evaluate_model(
    predictions: DataFrame,
    label_col: str,
    model_name: str
) -> dict:
    """
    Computes global RMSE, MAE, and R² on a predictions DataFrame.

    Parameters
    ----------
    predictions : Spark DataFrame with label and prediction columns
    label_col   : name of the ground truth column
    model_name  : label for printed output

    Returns
    -------
    dict with keys: model, rmse, mae, r2
    """
    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol="prediction"
    )

    rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    mae  = evaluator.setMetricName("mae").evaluate(predictions)
    r2   = evaluator.setMetricName("r2").evaluate(predictions)
    mape = predictions.withColumn(
        "absolute_percentage_error",
        F.when(F.col("total_active_energy_kwh") == 0, 0)
        .otherwise(F.abs(F.col("total_active_energy_kwh") - F.col("prediction")) / F.col("total_active_energy_kwh"))
    ).select(F.mean("absolute_percentage_error") * 100).collect()[0][0]

    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"R2:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'='*50}")

    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


def evaluate_by_municipality(
    predictions: DataFrame,
    label_col: str,
    model_name: str,
    plot: bool = True,
    plot_dir: str = "code/plots/",
    top_n: int = 20
) -> pd.DataFrame:
    """
    Computes per-municipality RMSE, MAE, R², and MAPE.

    Aggregates predictions in Spark (big-data-safe), then collects
    the small municipality-level result (~308 rows) for analysis
    and visualisation.

    Per-municipality metrics reveal:
      - Which municipalities the model struggles with most
      - Whether errors are systematic (high-consumption areas) or random
      - Whether is_holiday / is_weekend features help specific regions

    Parameters
    ----------
    predictions : Spark DataFrame with municipality, label, prediction columns
    label_col   : name of the ground truth column
    model_name  : label for plot titles
    plot        : whether to generate and save plots
    plot_dir    : directory to save plots
    top_n       : number of municipalities to show in bar charts

    Returns
    -------
    Pandas DataFrame with one row per municipality and columns:
        municipality, n_rows, rmse, mae, r2, mape, mean_actual, mean_predicted
    """

    os.makedirs(plot_dir, exist_ok=True)

    # Compute per-municipality metrics in Spark.
    #
    # We compute RMSE and MAE manually via aggregation rather than looping
    # RegressionEvaluator per municipality — one Spark job instead of 308.
    #
    # MAPE (Mean Absolute Percentage Error) is included as it is scale-invariant:
    # it allows fair comparison between high-consuming cities (Lisboa) and
    # low-consuming rural municipalities without the large municipalities
    # dominating the metric.
    #
    # NOTE: division by zero guard on MAPE — municipalities with zero actual
    # consumption (e.g. data gaps) are excluded from MAPE calculation.
    
    per_municipality = (
        predictions
        .withColumn("error",     F.col("prediction") - F.col(label_col))
        .withColumn("sq_error",  F.pow(F.col("prediction") - F.col(label_col), 2))
        .withColumn("abs_error", F.abs(F.col("prediction") - F.col(label_col)))
        .withColumn("abs_pct_error",
            F.when(F.col(label_col) != 0,
                F.abs(F.col("prediction") - F.col(label_col)) / F.abs(F.col(label_col)) * 100
            ).otherwise(None)
        )
        .groupBy("municipality")
        .agg(
            F.count("*")                        .alias("n_rows"),
            F.sqrt(F.avg("sq_error"))           .alias("rmse"),
            F.avg("abs_error")                  .alias("mae"),
            F.avg("abs_pct_error")              .alias("mape"),
            F.avg(label_col)                    .alias("mean_actual"),
            F.avg("prediction")                 .alias("mean_predicted"),
            F.avg("error")                      .alias("mean_bias"),  # positive = over-prediction
        )
        .withColumn("r2",
            # R² per municipality via aggregation
            # NOTE: requires a second pass — computed separately below
            F.lit(None).cast("double")
        )
        .orderBy("rmse", ascending=False)
    )

    # Compute per-municipality R² separately.
    #
    # R² = 1 - SS_res / SS_tot requires knowing the mean per municipality,
    # which requires a window function. We compute it in a second pass.

    munic_window = Window.partitionBy("municipality")

    r2_df = (
        predictions
        .withColumn("mean_actual",
            F.avg(label_col).over(munic_window))
        .withColumn("ss_res",
            F.pow(F.col(label_col) - F.col("prediction"), 2))
        .withColumn("ss_tot",
            F.pow(F.col(label_col) - F.col("mean_actual"), 2))
        .groupBy("municipality")
        .agg(
            (1 - F.sum("ss_res") / F.sum("ss_tot")).alias("r2")
        )
    )

    # Join R² back in
    per_municipality = (
        per_municipality
        .drop("r2")
        .join(r2_df, on="municipality", how="left")
    )

    # Collect to Pandas — safe because result is ~308 rows (one per municipality)
    # NOTE: toPandas() used here for analysis and plotting only.
    
    metrics_pd = per_municipality.toPandas()
    metrics_pd = metrics_pd.round(2).sort_values("rmse", ascending=False)

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Per-Municipality Evaluation — {model_name}")
    print(f"{'='*60}")
    print(f"  Municipalities evaluated : {len(metrics_pd)}")
    print(f"  Median RMSE              : {metrics_pd['rmse'].median():,.2f} kWh")
    print(f"  Median MAPE              : {metrics_pd['mape'].median():.2f}%")
    print(f"  Worst municipality (RMSE): {metrics_pd.iloc[0]['municipality']}  "
          f"({metrics_pd.iloc[0]['rmse']:,.2f} kWh)")
    print(f"  Best  municipality (RMSE): {metrics_pd.iloc[-1]['municipality']}  "
          f"({metrics_pd.iloc[-1]['rmse']:,.2f} kWh)")
    print(f"{'='*60}\n")
    print(metrics_pd.head(15).to_string(index=False))

    if not plot:
        return metrics_pd

    # ── Plot 1: RMSE by municipality (top N worst) ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    top_worst = metrics_pd.head(top_n)
    bars = ax.barh(
        top_worst["municipality"][::-1],
        top_worst["rmse"][::-1],
        color=sns.color_palette("Reds_r", top_n)
    )
    ax.set(
        title=f"Top {top_n} Municipalities by RMSE — {model_name}",
        xlabel="RMSE (kWh)", ylabel=None
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_by_municipality_{model_name.replace(' ', '_')}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # ── Plot 2: MAPE distribution ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(metrics_pd["mape"].dropna(), bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(metrics_pd["mape"].median(), color="tomato", linestyle="--",
               linewidth=2, label=f"Median MAPE: {metrics_pd['mape'].median():.1f}%")
    ax.set(
        title=f"MAPE Distribution Across Municipalities — {model_name}",
        xlabel="MAPE (%)", ylabel="Number of Municipalities"
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mape_distribution_{model_name.replace(' ', '_')}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # ── Plot 3: Mean bias — are we over or under predicting? ─────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    bias_sorted = metrics_pd.sort_values("mean_bias")
    colors = ["tomato" if b > 0 else "steelblue" for b in bias_sorted["mean_bias"]]
    ax.bar(range(len(bias_sorted)), bias_sorted["mean_bias"], color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title=f"Mean Prediction Bias by Municipality — {model_name}",
        xlabel="Municipality (sorted by bias)",
        ylabel="Mean Error (kWh)\nPositive = Over-prediction"
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"bias_by_municipality_{model_name.replace(' ', '_')}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # ── Plot 4: RMSE vs mean actual consumption ───────────────────────────────
    # Higher-consuming municipalities naturally have higher absolute RMSE.
    # This scatter checks whether errors are proportional (good) or
    # whether specific municipalities are outliers (bad — needs investigation).
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        metrics_pd["mean_actual"],
        metrics_pd["rmse"],
        alpha=0.6, color="steelblue", s=40
    )
    # Annotate worst outliers
    for _, row in metrics_pd.head(5).iterrows():
        ax.annotate(
            row["municipality"],
            xy=(row["mean_actual"], row["rmse"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7, color="tomato"
        )
    ax.set(
        title=f"RMSE vs Mean Consumption — {model_name}",
        xlabel="Mean Actual Consumption (kWh)",
        ylabel="RMSE (kWh)"
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_consumption_{model_name.replace(' ', '_')}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    return metrics_pd


def compare_models(
    metrics_list: list,
    plot_dir: str = "code/plots/"
) -> pd.DataFrame:
    """
    Produces a side-by-side comparison table and plot for multiple models.

    Parameters
    ----------
    metrics_list : list of dicts returned by evaluate_model()
    plot_dir     : directory to save comparison plot

    Returns
    -------
    Pandas DataFrame with one row per model
    """
    os.makedirs(plot_dir, exist_ok=True)

    comparison = pd.DataFrame(metrics_list).round(4)

    print("\nModel Comparison:")
    print(comparison.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, metric in zip(axes, ["rmse", "mae", "r2"]):
        sns.barplot(data=comparison, x="model", y=metric,
                    palette="Blues_r", ax=ax)
        ax.set(title=metric.upper(), xlabel=None)
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Model Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "model_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    return comparison