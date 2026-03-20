from pathlib import Path
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich import print
from scipy import stats

from negmas.gb.negotiators.timebased import (
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
)
from negmas.helpers import humanize_time
from negmas.helpers.strings import unique_name
from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.tournaments.neg import cartesian_tournament

from adaptive_agent import AdaptiveNegotiator
from microNegotiator import MicroNegotiator
from timeBasedAgent import TimeBasedAgent


def get_scenarios(n=10) -> list[Scenario]:
    issues = (
        make_issue([f"{i}" for i in range(10)], "quantity"),
        make_issue([f"{i}" for i in range(5)], "price"),
    )

    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.6), normalized=True),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=True),
        )
        for _ in range(n)
    ]

    return [
        Scenario(outcome_space=make_os(issues, name=f"S{i}"), ufuns=u)
        for i, u in enumerate(ufuns)
    ]


def plot_kde(results):
    fig = go.Figure()
    strategies = results.scores["strategy"].unique()

    for strategy in strategies:
        data = (
            results.scores[results.scores["strategy"] == strategy]["advantage"]
            .dropna()
            .values
        )
        if len(data) > 1:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
            fig.add_trace(
                go.Scatter(x=x_range, y=kde(x_range), mode="lines", name=str(strategy))
            )

    fig.update_layout(
        title="Advantage Distribution by Strategy",
        xaxis_title="Advantage",
        yaxis_title="Density",
    )
    fig.show()


def print_extensive_evaluation(scores):
    df = scores.copy()

    # ── Agreement detection ──────────────────────────────────────────────────
    # In NegMAS a failed negotiation returns the reserved value as utility,
    # so advantage ≈ 0 means no deal.
    df["agreed"] = df["advantage"].fillna(0) > 1e-6

    strategies = df["strategy"].unique()

    # ── 1. Overall Scores ────────────────────────────────────────────────────
    print(
        "\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]"
    )
    print("[bold cyan]          EXTENSIVE EVALUATION REPORT             [/bold cyan]")
    print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")

    # ── 2. Agreement & Error Rates ───────────────────────────────────────────
    print(
        "\n[bold yellow]── Agreement & Error Rates ─────────────────────────[/bold yellow]"
    )
    agg = (
        df.groupby("strategy")
        .agg(
            total=("agreed", "count"),
            agreements=("agreed", "sum"),
            errors=("has_error", "sum"),
        )
        .assign(
            agreement_rate=lambda x: x["agreements"] / x["total"] * 100,
            error_rate=lambda x: x["errors"] / x["total"] * 100,
        )
        .sort_values("agreement_rate", ascending=False)
    )
    agg["agreement_rate"] = agg["agreement_rate"].map("{:.1f}%".format)
    agg["error_rate"] = agg["error_rate"].map("{:.1f}%".format)
    print(
        agg[
            ["total", "agreements", "agreement_rate", "errors", "error_rate"]
        ].to_string()
    )

    # ── 3. Utility & Welfare ─────────────────────────────────────────────────
    print(
        "\n[bold yellow]── Utility & Welfare (agreements only) ─────────────[/bold yellow]"
    )
    agreed = df[df["agreed"]]
    uw = (
        agreed.groupby("strategy")[
            ["utility", "advantage", "partner_welfare", "welfare", "fairness"]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    # Flatten for readability
    uw.columns = ["_".join(c) for c in uw.columns]
    print(
        uw[
            [
                "utility_mean",
                "utility_std",
                "advantage_mean",
                "welfare_mean",
                "fairness_mean",
                "fairness_std",
            ]
        ]
        .sort_values("utility_mean", ascending=False)
        .to_string()
    )

    # ── 4. Optimality Metrics ────────────────────────────────────────────────
    print(
        "\n[bold yellow]── Optimality Metrics (agreements only) ────────────[/bold yellow]"
    )
    opt_cols = [
        c
        for c in [
            "nash_optimality",
            "kalai_optimality",
            "ks_optimality",
            "modified_kalai_optimality",
            "modified_ks_optimality",
            "max_welfare_optimality",
            "pareto_optimality",
        ]
        if c in agreed.columns
    ]
    opt = agreed.groupby("strategy")[opt_cols].mean().round(4)
    opt = opt.sort_values("nash_optimality", ascending=False)
    print(opt.to_string())

    # ── 5. Time Statistics ───────────────────────────────────────────────────
    if "time" in df.columns:
        print(
            "\n[bold yellow]── Negotiation Time Statistics ─────────────────────[/bold yellow]"
        )
        time_stats = (
            df.groupby("strategy")["time"]
            .agg(mean="mean", std="std", min="min", max="max")
            .round(4)
            .sort_values("mean")
        )
        print(time_stats.to_string())

    # ── 6. Per-Scenario Breakdown ────────────────────────────────────────────
    print(
        "\n[bold yellow]── Per-Scenario Agreement Rate ──────────────────────[/bold yellow]"
    )
    scen = (
        df.groupby(["scenario", "strategy"])["agreed"]
        .mean()
        .mul(100)
        .round(1)
        .unstack(level="strategy")
        .fillna(0)
    )
    print(scen.to_string())

    # ── 7. Rank Summary ──────────────────────────────────────────────────────
    print(
        "\n[bold yellow]── Overall Rank Summary ─────────────────────────────[/bold yellow]"
    )
    rank_df = pd.DataFrame(index=strategies)
    rank_df["utility"] = agreed.groupby("strategy")["utility"].mean()
    rank_df["advantage"] = agreed.groupby("strategy")["advantage"].mean()
    rank_df["nash_opt"] = (
        agreed.groupby("strategy")["nash_optimality"].mean()
        if "nash_optimality" in agreed.columns
        else np.nan
    )
    rank_df["fairness"] = (
        agreed.groupby("strategy")["fairness"].mean()
        if "fairness" in agreed.columns
        else np.nan
    )
    rank_df["agree_%"] = df.groupby("strategy")["agreed"].mean() * 100
    # Composite: normalise each column then average
    norm = (rank_df - rank_df.min()) / (rank_df.max() - rank_df.min() + 1e-12)
    rank_df["composite"] = norm.mean(axis=1)
    rank_df = rank_df.sort_values("composite", ascending=False).round(4)
    for col in ["agree_%"]:
        rank_df[col] = rank_df[col].map("{:.1f}%".format)
    print(rank_df.to_string())
    print()


def main():
    tic = time.perf_counter()
    path = Path.home() / "negmas" / unique_name("group31_test")

    results = cartesian_tournament(
        competitors=[
            AdaptiveNegotiator,
            BoulwareTBNegotiator,
            ConcederTBNegotiator,
            LinearTBNegotiator,
            MicroNegotiator,
            TimeBasedAgent,
        ],
        scenarios=get_scenarios(n=10),
        n_repetitions=20,
        path=path,
        njobs=-1,  # Serialaize
    )

    print("\n[bold]Scores summary[/bold]")
    print(results.scores_summary)

    print_extensive_evaluation(results.scores)

    print(f"\nDone in {humanize_time(time.perf_counter() - tic)}")

    plot_kde(results)


if __name__ == "__main__":
    mp.freeze_support()
    main()
