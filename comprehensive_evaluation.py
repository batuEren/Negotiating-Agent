"""
comprehensive_evaluation.py
===========================
Comprehensive evaluation of AdaptivePrONeg vs MicroNegotiator.

Scenarios
---------
1. Single-Issue Price        — tight ZOPA (3 viable prices out of 13)
2. Multi-Issue Balanced      — 3 issues, compatible on quantity; moderate ZOPA
3. Multi-Issue High Conflict — same 3 issues, higher reservation values; tight ZOPA

Agent combinations
------------------
  Adaptive (buyer) vs Adaptive (seller)
  Adaptive (buyer) vs Micro   (seller)
  Micro    (buyer) vs Adaptive (seller)
  Micro    (buyer) vs Micro    (seller)

Metrics (per agreed negotiation unless stated)
----------------------------------------------
  agreement_rate    — fraction of runs that reached agreement   (all runs)
  buyer_utility     — buyer's utility at agreement
  seller_utility    — seller's utility at agreement
  social_welfare    — buyer_utility + seller_utility
  nash_efficiency   — nash_product / max_possible_nash_product  (0=worst, 1=optimal)
  pareto_distance   — Euclidean distance from agreement to Pareto frontier
  fairness          — |buyer_utility - seller_utility|          (lower = more equal)
  n_rounds          — negotiation steps taken                   (all runs)

Plots
-----
  1. Agreement Rate              — bar chart per scenario
  2. Utility Distributions       — box plots (buyer + seller) per scenario
  3. Social Welfare & Nash       — bar charts per scenario
  4. Pareto Efficiency           — bar chart per scenario
  5. Negotiation Length          — box plot per scenario
  6. Utility Space               — scatter + Pareto + Nash per scenario
"""

from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from negmas import make_issue, SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun
from rich.console import Console
from rich.table import Table

from adaptive_proneg import AdaptivePrONeg
from microNegotiator import MicroNegotiator

warnings.filterwarnings("ignore")
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Agent colour palette
# ─────────────────────────────────────────────────────────────────────────────
COMBO_LABELS = {
    ("AdaptivePrONeg", "AdaptivePrONeg"): "Adap vs Adap",
    ("AdaptivePrONeg", "MicroNegotiator"):    "Adap vs Micro",
    ("MicroNegotiator",    "AdaptivePrONeg"): "Micro vs Adap",
    ("MicroNegotiator",    "MicroNegotiator"):    "Micro vs Micro",
}
COMBO_ORDER = [
    "Adap vs Adap",
    "Adap vs Micro",
    "Micro vs Adap",
    "Micro vs Micro",
]
COLORS = {
    "Adap vs Adap":  "#2196F3",
    "Adap vs Micro": "#4CAF50",
    "Micro vs Adap": "#FF9800",
    "Micro vs Micro":"#F44336",
}
COMBOS = list(COMBO_LABELS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

def scenario_single_issue(n_steps: int = 30):
    """
    Single-issue price negotiation.
    Price ∈ {0..12} (13 discrete values).
    Seller reservation: price ≥ 6  → rv ≈ 0.50
    Buyer  reservation: price ≤ 8  → rv ≈ 0.33
    ZOPA: prices 6, 7, 8  (3 outcomes).
    """
    issues  = [make_issue(name="price", values=13)]
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    seller_ufun = LUFun(
        values={"price": IdentityFun()},
        outcome_space=session.outcome_space,
        reserved_value=6 / 12,
    ).scale_max(1.0)

    buyer_ufun = LUFun(
        values={"price": AffineFun(-1, bias=12.0)},
        outcome_space=session.outcome_space,
        reserved_value=(12 - 8) / 12,
    ).scale_max(1.0)

    return session, seller_ufun, buyer_ufun


def scenario_multi_issue_balanced(n_steps: int = 30):
    """
    Three-issue negotiation: price (0-9), quantity (1-11), delivery_time (0-9).
    Seller: high price, high delivery speed (low time), moderate quantity.
    Buyer:  low  price, low  delivery speed (high time), moderate quantity.
    Both benefit from quantity → good integrative potential.
    Reservation values: seller 0.40, buyer 0.20.
    """
    issues = [
        make_issue(name="price",         values=10),
        make_issue(name="quantity",      values=(1, 11)),
        make_issue(name="delivery_time", values=10),
    ]
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    seller_ufun = LUFun(
        values={
            "price":         IdentityFun(),
            "quantity":      LinearFun(0.2),
            "delivery_time": AffineFun(-1, bias=9.0),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.40,
    ).scale_max(1.0)

    buyer_ufun = LUFun(
        values={
            "price":         AffineFun(-1, bias=9.0),
            "quantity":      LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.20,
    ).scale_max(1.0)

    return session, seller_ufun, buyer_ufun


def scenario_multi_issue_high_conflict(n_steps: int = 30):
    """
    Same three-issue structure as balanced, but with significantly higher
    reservation values (seller 0.60, buyer 0.50).
    Fewer outcomes fall inside the ZOPA → tests agents under deadline pressure
    and exploits AdaptivePrONeg's PrONeg low-agreement-probability logic.
    """
    issues = [
        make_issue(name="price",         values=10),
        make_issue(name="quantity",      values=(1, 11)),
        make_issue(name="delivery_time", values=10),
    ]
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    seller_ufun = LUFun(
        values={
            "price":         IdentityFun(),
            "quantity":      LinearFun(0.2),
            "delivery_time": AffineFun(-1, bias=9.0),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.60,
    ).scale_max(1.0)

    buyer_ufun = LUFun(
        values={
            "price":         AffineFun(-1, bias=9.0),
            "quantity":      LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.50,
    ).scale_max(1.0)

    return session, seller_ufun, buyer_ufun


SCENARIOS: dict[str, callable] = {
    "Single-Issue Price":        scenario_single_issue,
    "Multi-Issue Balanced":      scenario_multi_issue_balanced,
    "Multi-Issue High Conflict": scenario_multi_issue_high_conflict,
}


# ─────────────────────────────────────────────────────────────────────────────
# Outcome-space helpers  (precomputed once per scenario)
# ─────────────────────────────────────────────────────────────────────────────

def _enumerate_outcomes(session) -> list[tuple]:
    if hasattr(session, "outcomes") and session.outcomes is not None:
        try:
            return [tuple(o) for o in session.outcomes]
        except Exception:
            pass
    os_ = getattr(session, "outcome_space", None)
    if os_ is not None:
        for attr_name in ("enumerate_or_sample", "enumerate", "all"):
            if hasattr(os_, attr_name):
                attr = getattr(os_, attr_name)
                try:
                    vals = attr() if callable(attr) else attr
                    return [tuple(o) for o in vals]
                except Exception:
                    continue
    return []


def _pareto_frontier(outcomes, bu_dict, su_dict) -> list[tuple]:
    """
    Compute Pareto-optimal outcomes (maximise both buyer and seller utility).
    O(n²) — fast enough for |outcomes| ≤ 1100.
    Returns list of (outcome, bu, su).
    """
    pts = [(o, bu_dict[o], su_dict[o]) for o in outcomes]
    pareto = []
    for i, (oi, bi, si) in enumerate(pts):
        dominated = any(
            bj >= bi and sj >= si and (bj > bi or sj > si)
            for j, (oj, bj, sj) in enumerate(pts) if i != j
        )
        if not dominated:
            pareto.append((oi, bi, si))
    return pareto


def _nash_point(pareto, buyer_rv, seller_rv):
    """Nash bargaining solution: argmax (u_b - rv_b)(u_s - rv_s) on Pareto frontier."""
    best, best_val = None, -1.0
    for o, bu, su in pareto:
        val = max(0.0, bu - buyer_rv) * max(0.0, su - seller_rv)
        if val > best_val:
            best_val, best = val, (o, bu, su)
    return best, best_val  # (point, max_nash_product)


def precompute_scenario(scenario_fn, n_steps: int = 30) -> dict:
    """
    Run scenario once to enumerate outcomes and compute static analysis objects.
    Results are reused across all runs of the same scenario to avoid
    redundant O(n²) Pareto computations.
    """
    session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
    outcomes = _enumerate_outcomes(session)

    bu_dict  = {o: float(buyer_ufun(o))  for o in outcomes}
    su_dict  = {o: float(seller_ufun(o)) for o in outcomes}

    buyer_rv  = float(buyer_ufun.reserved_value  or 0.0)
    seller_rv = float(seller_ufun.reserved_value or 0.0)

    pareto         = _pareto_frontier(outcomes, bu_dict, su_dict)
    nash_pt, max_nash = _nash_point(pareto, buyer_rv, seller_rv)

    return {
        "outcomes":  outcomes,
        "bu_dict":   bu_dict,
        "su_dict":   su_dict,
        "pareto":    pareto,
        "nash_pt":   nash_pt,
        "max_nash":  max(max_nash, 1e-9),
        "buyer_rv":  buyer_rv,
        "seller_rv": seller_rv,
    }


def _pareto_distance(bu: float, su: float, pareto: list[tuple]) -> float:
    if not pareto:
        return float("nan")
    return min(math.sqrt((bu - p_bu) ** 2 + (su - p_su) ** 2)
               for _, p_bu, p_su in pareto)


# ─────────────────────────────────────────────────────────────────────────────
# Single negotiation run
# ─────────────────────────────────────────────────────────────────────────────

def _make_agent(agent_type: str, role: str):
    if agent_type == "AdaptivePrONeg":
        return AdaptivePrONeg(name=role)
    return MicroNegotiator(name=role)


def run_one(scenario_fn, buyer_type: str, seller_type: str,
            n_steps: int, pre: dict) -> dict:
    session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
    session.add(_make_agent(buyer_type,  "buyer"),  ufun=buyer_ufun)
    session.add(_make_agent(seller_type, "seller"), ufun=seller_ufun)
    result = session.run()

    agreement = getattr(result, "agreement", None)
    if agreement is not None:
        agreement = tuple(agreement)
    agreed = agreement is not None

    buyer_rv  = pre["buyer_rv"]
    seller_rv = pre["seller_rv"]

    if agreed:
        bu = pre["bu_dict"].get(agreement, float(buyer_ufun(agreement)))
        su = pre["su_dict"].get(agreement, float(seller_ufun(agreement)))
        nash_prod     = max(0.0, bu - buyer_rv) * max(0.0, su - seller_rv)
        nash_eff      = nash_prod / pre["max_nash"]
        pareto_dist   = _pareto_distance(bu, su, pre["pareto"])
        fairness      = abs(bu - su)
    else:
        bu = buyer_rv
        su = seller_rv
        nash_prod   = float("nan")
        nash_eff    = float("nan")
        pareto_dist = float("nan")
        fairness    = float("nan")

    n_rounds = len(session.history) if getattr(session, "history", None) else 0
    combo    = COMBO_LABELS.get((buyer_type, seller_type),
                                f"{buyer_type} vs {seller_type}")
    return {
        "buyer_type":     buyer_type,
        "seller_type":    seller_type,
        "combo":          combo,
        "agreed":         agreed,
        "agreement":      agreement,
        "buyer_utility":  bu,
        "seller_utility": su,
        "social_welfare": bu + su,
        "nash_product":   nash_prod,
        "nash_efficiency":nash_eff,
        "pareto_distance":pareto_dist,
        "fairness":       fairness,
        "n_rounds":       n_rounds,
        "buyer_rv":       buyer_rv,
        "seller_rv":      seller_rv,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario_evaluation(scenario_name: str, scenario_fn,
                             n_runs: int = 30, n_steps: int = 30) -> pd.DataFrame:
    console.print(f"\n[bold cyan]▶  {scenario_name}[/bold cyan]  "
                  f"({n_runs} runs × {len(COMBOS)} combos)")

    pre  = precompute_scenario(scenario_fn, n_steps=n_steps)
    rows = []
    for run in range(n_runs):
        for buyer_type, seller_type in COMBOS:
            row           = run_one(scenario_fn, buyer_type, seller_type, n_steps, pre)
            row["run"]      = run
            row["scenario"] = scenario_name
            rows.append(row)
        if (run + 1) % 10 == 0:
            console.print(f"   [{run + 1}/{n_runs}]")

    df = pd.DataFrame(rows)

    # Attach scenario-level reference data for plotting
    df.attrs["buyer_rv"]  = pre["buyer_rv"]
    df.attrs["seller_rv"] = pre["seller_rv"]
    df.attrs["pareto"]    = pre["pareto"]
    df.attrs["nash_pt"]   = pre["nash_pt"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rich console summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_scenario_stats(df: pd.DataFrame, scenario_name: str):
    console.print(f"\n[bold yellow]── {scenario_name} ──[/bold yellow]")

    tbl = Table(show_header=True, header_style="bold magenta")
    tbl.add_column("Combination",    style="cyan", no_wrap=True)
    tbl.add_column("Agree %",        justify="right")
    tbl.add_column("Buyer Util",     justify="right")
    tbl.add_column("Seller Util",    justify="right")
    tbl.add_column("Social Welfare", justify="right")
    tbl.add_column("Nash Effic.",    justify="right")
    tbl.add_column("Pareto Dist",    justify="right")
    tbl.add_column("Fairness",       justify="right")
    tbl.add_column("Rounds",         justify="right")

    def _fmt(series, dec=3):
        vals = series.dropna()
        if vals.empty:
            return "[red]-[/red]"
        return f"{vals.mean():.{dec}f} ±{vals.std():.{dec}f}"

    for combo in COMBO_ORDER:
        grp    = df[df["combo"] == combo]
        agreed = grp[grp["agreed"]]
        rate   = 100 * grp["agreed"].mean()

        tbl.add_row(
            combo,
            f"[green]{rate:.0f}%[/green]" if rate >= 80 else
            (f"[yellow]{rate:.0f}%[/yellow]" if rate >= 40 else f"[red]{rate:.0f}%[/red]"),
            _fmt(agreed["buyer_utility"])  if not agreed.empty else "[red]-[/red]",
            _fmt(agreed["seller_utility"]) if not agreed.empty else "[red]-[/red]",
            _fmt(agreed["social_welfare"]) if not agreed.empty else "[red]-[/red]",
            _fmt(agreed["nash_efficiency"])if not agreed.empty else "[red]-[/red]",
            _fmt(agreed["pareto_distance"])if not agreed.empty else "[red]-[/red]",
            _fmt(agreed["fairness"])       if not agreed.empty else "[red]-[/red]",
            _fmt(grp["n_rounds"], 1),
        )
    console.print(tbl)


def print_overall_summary(combined: pd.DataFrame):
    console.print("\n[bold white on blue] OVERALL SUMMARY (all scenarios combined) [/bold white on blue]")

    tbl = Table(show_header=True, header_style="bold magenta")
    tbl.add_column("Combination",    style="cyan", no_wrap=True)
    tbl.add_column("Agree %",        justify="right")
    tbl.add_column("Social Welfare", justify="right")
    tbl.add_column("Nash Effic.",    justify="right")
    tbl.add_column("Pareto Dist",    justify="right")

    for combo in COMBO_ORDER:
        grp    = combined[combined["combo"] == combo]
        agreed = grp[grp["agreed"]]
        rate   = 100 * grp["agreed"].mean()

        def _f(col, dec=3):
            v = agreed[col].dropna()
            return f"{v.mean():.{dec}f}" if not v.empty else "-"

        tbl.add_row(
            combo,
            f"{rate:.1f}%",
            _f("social_welfare"),
            _f("nash_efficiency"),
            _f("pareto_distance"),
        )
    console.print(tbl)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _combo_colors(combos):
    return [COLORS.get(c, "#888888") for c in combos]


def _ordered_combos(df):
    present = set(df["combo"].unique())
    return [c for c in COMBO_ORDER if c in present]


# ── Plot 1: Agreement Rates ────────────────────────────────────────────────

def plot_agreement_rates(all_dfs: list, scenario_names: list):
    n   = len(scenario_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, zip(scenario_names, all_dfs)):
        combos = _ordered_combos(df)
        rates  = [df[df["combo"] == c]["agreed"].mean() * 100 for c in combos]
        colors = _combo_colors(combos)
        bars   = ax.bar(range(len(combos)), rates, color=colors,
                        edgecolor="black", linewidth=0.6, width=0.6)
        ax.set_title(name, fontweight="bold", fontsize=10)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Agreement Rate (%)" if ax is axes[0] else "")
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Agreement Rate by Scenario & Agent Combination",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()


# ── Plot 2: Utility Box Plots ──────────────────────────────────────────────

def plot_utility_boxplots(all_dfs: list, scenario_names: list):
    n   = len(scenario_names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (name, df) in enumerate(zip(scenario_names, all_dfs)):
        agreed = df[df["agreed"]]
        combos = _ordered_combos(df)

        for row, (metric, label) in enumerate([
            ("buyer_utility",  "Buyer Utility at Agreement"),
            ("seller_utility", "Seller Utility at Agreement"),
        ]):
            ax   = axes[row][col]
            data = [agreed[agreed["combo"] == c][metric].dropna().values
                    for c in combos]
            bp   = ax.boxplot(data, patch_artist=True, widths=0.5,
                              medianprops={"color": "black", "linewidth": 2})
            for patch, combo in zip(bp["boxes"], combos):
                patch.set_facecolor(COLORS.get(combo, "#888"))
                patch.set_alpha(0.75)

            # Reservation value reference line
            rv_col = "buyer_rv" if row == 0 else "seller_rv"
            if not agreed.empty:
                rv = agreed[rv_col].iloc[0]
                ax.axhline(rv, color="red", linestyle="--", linewidth=1.2,
                           label=f"Reservation ({rv:.2f})")
                ax.legend(fontsize=7, loc="lower right")

            ax.set_title(f"{name}\n{label}", fontsize=9, fontweight="bold")
            ax.set_xticks(range(1, len(combos) + 1))
            ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Utility")
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Utility Distributions at Agreement", fontweight="bold", fontsize=12)
    plt.tight_layout()


# ── Plot 3: Social Welfare & Nash Efficiency ───────────────────────────────

def plot_welfare_and_nash(all_dfs: list, scenario_names: list):
    n   = len(scenario_names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (name, df) in enumerate(zip(scenario_names, all_dfs)):
        agreed = df[df["agreed"]]
        combos = _ordered_combos(df)
        colors = _combo_colors(combos)

        for row, (metric, ylabel, ymax) in enumerate([
            ("social_welfare",  "Social Welfare (u_buyer + u_seller)", 2.0),
            ("nash_efficiency", "Nash Efficiency (0=worst, 1=optimal)", 1.1),
        ]):
            ax    = axes[row][col]
            means = [agreed[agreed["combo"] == c][metric].mean() for c in combos]
            stds  = [agreed[agreed["combo"] == c][metric].std()  for c in combos]

            bars = ax.bar(range(len(combos)), means, yerr=stds,
                          color=colors, edgecolor="black", linewidth=0.6,
                          error_kw={"elinewidth": 1.5, "capsize": 4},
                          capsize=4, width=0.6)
            if row == 0:
                ax.set_title(name, fontweight="bold", fontsize=10)
            ax.set_xticks(range(len(combos)))
            ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, ymax)
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Social Welfare & Nash Efficiency", fontweight="bold", fontsize=12)
    plt.tight_layout()


# ── Plot 4: Pareto Distance ────────────────────────────────────────────────

def plot_pareto_efficiency(all_dfs: list, scenario_names: list):
    n   = len(scenario_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, zip(scenario_names, all_dfs)):
        agreed = df[df["agreed"]]
        combos = _ordered_combos(df)
        means  = [agreed[agreed["combo"] == c]["pareto_distance"].mean()
                  for c in combos]
        stds   = [agreed[agreed["combo"] == c]["pareto_distance"].std()
                  for c in combos]
        colors = _combo_colors(combos)

        ax.bar(range(len(combos)), means, yerr=stds, color=colors,
               edgecolor="black", linewidth=0.6,
               error_kw={"elinewidth": 1.5, "capsize": 4}, width=0.6)
        ax.set_title(name, fontweight="bold", fontsize=10)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Pareto Distance (lower = better)"
                      if ax is axes[0] else "")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Pareto Efficiency of Agreements (lower = closer to frontier)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()


# ── Plot 5: Rounds to Agreement ────────────────────────────────────────────

def plot_negotiation_length(all_dfs: list, scenario_names: list):
    n   = len(scenario_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, zip(scenario_names, all_dfs)):
        combos = _ordered_combos(df)
        data   = [df[df["combo"] == c]["n_rounds"].values for c in combos]
        bp     = ax.boxplot(data, patch_artist=True, widths=0.5,
                            medianprops={"color": "black", "linewidth": 2})
        for patch, combo in zip(bp["boxes"], combos):
            patch.set_facecolor(COLORS.get(combo, "#888"))
            patch.set_alpha(0.75)
        ax.set_title(name, fontweight="bold", fontsize=10)
        ax.set_xticks(range(1, len(combos) + 1))
        ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Rounds" if ax is axes[0] else "")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Negotiation Length (all runs)", fontweight="bold", fontsize=12)
    plt.tight_layout()


# ── Plot 6: Utility Space per Scenario ────────────────────────────────────

def plot_utility_space(df: pd.DataFrame, scenario_name: str):
    """
    Scatter of all agreement points in utility space, overlaid with the
    full outcome cloud, Pareto frontier, Nash point, and reservation lines.
    """
    pareto  = df.attrs.get("pareto",   [])
    nash_pt = df.attrs.get("nash_pt",  None)
    buyer_rv  = df.attrs.get("buyer_rv",  0.0)
    seller_rv = df.attrs.get("seller_rv", 0.0)

    fig, ax = plt.subplots(figsize=(8, 7))

    # All outcomes (background cloud)
    if pareto:
        all_bu = [p[1] for p in pareto]
        all_su = [p[2] for p in pareto]
        ax.scatter(all_bu, all_su, s=40, color="steelblue", alpha=0.6,
                   zorder=3, label="Pareto frontier")
        # Connect frontier
        frontier_sorted = sorted(pareto, key=lambda x: x[1])
        ax.plot([p[1] for p in frontier_sorted],
                [p[2] for p in frontier_sorted],
                color="steelblue", linewidth=1.5, alpha=0.6)

    # Nash point
    if nash_pt:
        ax.scatter([nash_pt[1]], [nash_pt[2]], s=220, marker="D",
                   color="gold", edgecolor="black", zorder=6,
                   label=f"Nash ({nash_pt[1]:.2f}, {nash_pt[2]:.2f})")
        ax.annotate("Nash", (nash_pt[1], nash_pt[2]),
                    xytext=(8, -14), textcoords="offset points", fontsize=9)

    # Agreement points per combo
    for combo in _ordered_combos(df):
        grp = df[(df["combo"] == combo) & df["agreed"]]
        if grp.empty:
            continue
        ax.scatter(grp["buyer_utility"], grp["seller_utility"],
                   s=70, alpha=0.7, color=COLORS.get(combo, "#888"),
                   label=f"{combo} (n={len(grp)})", zorder=5,
                   edgecolors="black", linewidths=0.4)

    # Reservation lines
    ax.axvline(buyer_rv,  color="red",     linestyle="--", linewidth=1.5,
               label=f"Buyer rv = {buyer_rv:.2f}")
    ax.axhline(seller_rv, color="darkred", linestyle="--", linewidth=1.5,
               label=f"Seller rv = {seller_rv:.2f}")

    # ZOPA shading
    ax.axvspan(buyer_rv,  1.05, alpha=0.04, color="green")
    ax.axhspan(seller_rv, 1.05, alpha=0.04, color="green")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Buyer Utility", fontsize=11)
    ax.set_ylabel("Seller Utility", fontsize=11)
    ax.set_title(f"Utility Space — {scenario_name}", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()


# ── Plot 7: Comprehensive overview heatmap ─────────────────────────────────

def plot_heatmap_summary(combined: pd.DataFrame, scenario_names: list):
    """
    Heatmap: rows = agent combinations, cols = scenarios,
    cell = agreement rate (%) with colour coding.
    """
    data = {}
    for sname in scenario_names:
        col = {}
        sdf = combined[combined["scenario"] == sname]
        for combo in COMBO_ORDER:
            grp = sdf[sdf["combo"] == combo]
            col[combo] = grp["agreed"].mean() * 100 if not grp.empty else 0.0
        data[sname] = col

    matrix = pd.DataFrame(data, index=COMBO_ORDER)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Agreement Rate (%)")

    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(COMBO_ORDER)))
    ax.set_yticklabels(COMBO_ORDER, fontsize=9)

    for i in range(len(COMBO_ORDER)):
        for j in range(len(scenario_names)):
            val = matrix.iloc[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=10,
                    fontweight="bold",
                    color="white" if val < 40 or val > 80 else "black")

    ax.set_title("Agreement Rate (%) — Heatmap", fontweight="bold", fontsize=12)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    N_RUNS  = 30
    N_STEPS = 30

    console.rule("[bold green]Comprehensive Negotiation Evaluation[/bold green]")
    console.print(f"  Scenarios  : {list(SCENARIOS.keys())}")
    console.print(f"  Combos     : {len(COMBOS)}")
    console.print(f"  Runs/combo : {N_RUNS}  |  Max steps: {N_STEPS}")
    console.print(f"  Total runs : {len(SCENARIOS) * len(COMBOS) * N_RUNS}")

    all_dfs        = []
    scenario_names = list(SCENARIOS.keys())

    for sname, sfn in SCENARIOS.items():
        df = run_scenario_evaluation(sname, sfn, n_runs=N_RUNS, n_steps=N_STEPS)
        all_dfs.append(df)
        print_scenario_stats(df, sname)

    combined = pd.concat(all_dfs, ignore_index=True)
    print_overall_summary(combined)

    # ── Plots ──────────────────────────────────────────────────────────────
    console.print("\n[bold green]Generating plots…[/bold green]")

    plot_agreement_rates(all_dfs, scenario_names)
    plot_utility_boxplots(all_dfs, scenario_names)
    plot_welfare_and_nash(all_dfs, scenario_names)
    plot_pareto_efficiency(all_dfs, scenario_names)
    plot_negotiation_length(all_dfs, scenario_names)
    plot_heatmap_summary(combined, scenario_names)

    for sname, df in zip(scenario_names, all_dfs):
        plot_utility_space(df, sname)

    plt.show()
    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
