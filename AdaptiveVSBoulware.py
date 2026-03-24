"""
AdaptiveVSBoulware.py
=====================
Head-to-head comparison: AdaptivePrONeg vs BoulwareTBNegotiator.

Scenarios  : single-issue price  +  3-issue (price, quantity, delivery_time)
Evaluation : distance metrics (lower = better, 0 = at solution point),
             rank summary addressing the three assignment criteria (Sec 2.4):
               - Pareto frontier closeness  → pareto_dist
               - Nash Product closeness     → nash_dist
               - Social Welfare             → social_welfare

Plots
-----
  1. Utility Space          — agreements vs Pareto frontier & Nash point
  2. Concession Paths       — how offers trace through utility space over time
  3. Self vs Joint Trade-off— per-run scatter of (advantage, social_welfare)
  4. Radar Summary          — all 5 metrics at once for every combo
"""

from __future__ import annotations

import math
import time
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from negmas import make_issue, SAOMechanism
from negmas.gb.negotiators.timebased import BoulwareTBNegotiator
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun

from adaptive_proneg import AdaptivePrONeg


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────


def create_single_issue_scenario(n_steps: int = 30):
    """Price in {0..12}. Seller wants high, buyer wants low. ZOPA: 6–8."""
    issues = [make_issue(name="price", values=13)]
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    seller_ufun = LUFun(
        values={"price": IdentityFun()},
        outcome_space=session.outcome_space,
        reserved_value=6 / 12.0,
    ).scale_max(1.0)
    buyer_ufun = LUFun(
        values={"price": AffineFun(-1, bias=12.0)},
        outcome_space=session.outcome_space,
        reserved_value=(12.0 - 8) / 12.0,
    ).scale_max(1.0)
    return session, seller_ufun, buyer_ufun


def create_multi_issue_scenario(n_steps: int = 30):
    """3-issue: price, quantity, delivery_time. Opposing preferences on price
    and delivery; both prefer high quantity."""
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=(1, 11)),
        make_issue(name="delivery_time", values=10),
    ]
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    seller_ufun = LUFun(
        values={
            "price": IdentityFun(),
            "quantity": LinearFun(0.2),
            "delivery_time": AffineFun(-1, bias=9.0),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.4,
    ).scale_max(1.0)
    buyer_ufun = LUFun(
        values={
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.2,
    ).scale_max(1.0)
    return session, seller_ufun, buyer_ufun


SCENARIOS = {
    "Single-Issue Price": create_single_issue_scenario,
    "Multi-Issue (3 issues)": create_multi_issue_scenario,
}

COMBOS = [
    ("AdaptivePrONeg", "AdaptivePrONeg"),
    ("AdaptivePrONeg", "BoulwareTBNegotiator"),
    ("BoulwareTBNegotiator", "AdaptivePrONeg"),
    ("BoulwareTBNegotiator", "BoulwareTBNegotiator"),
]

COLORS = {
    "AdaptivePrONeg vs AdaptivePrONeg": "#2196F3",
    "AdaptivePrONeg vs BoulwareTBNegotiator": "#4CAF50",
    "BoulwareTBNegotiator vs AdaptivePrONeg": "#FF9800",
    "BoulwareTBNegotiator vs BoulwareTBNegotiator": "#F44336",
}

COMBO_SHORT = {
    "AdaptivePrONeg vs AdaptivePrONeg": "Adap/Adap",
    "AdaptivePrONeg vs BoulwareTBNegotiator": "Adap/Boul",
    "BoulwareTBNegotiator vs AdaptivePrONeg": "Boul/Adap",
    "BoulwareTBNegotiator vs BoulwareTBNegotiator": "Boul/Boul",
}


# ─────────────────────────────────────────────────────────────────────────────
# Outcome-space helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_outcomes(session) -> list:
    if hasattr(session, "outcomes") and session.outcomes is not None:
        try:
            return [tuple(o) for o in session.outcomes]
        except Exception:
            pass
    os_ = getattr(session, "outcome_space", None)
    if os_ is not None:
        for name in ("enumerate_or_sample", "enumerate", "all"):
            attr = getattr(os_, name, None)
            if attr is not None:
                try:
                    vals = attr() if callable(attr) else attr
                    return [tuple(o) for o in vals]
                except Exception:
                    continue
    return []


def _pareto_frontier(outcomes, buyer_ufun, seller_ufun):
    pts = sorted(
        [(o, float(buyer_ufun(o)), float(seller_ufun(o))) for o in outcomes],
        key=lambda x: (-x[1], x[2]),
    )
    pareto, max_su = [], -math.inf
    i = 0
    while i < len(pts):
        bu = pts[i][1]
        j = i
        while j < len(pts) and pts[j][1] == bu:
            j += 1
        group = pts[i:j]
        best_su = max(x[2] for x in group)
        if best_su > max_su:
            pareto.extend(p for p in group if p[2] == best_su)
            max_su = best_su
        i = j
    return pareto


def _nash_point(pareto, buyer_rv: float, seller_rv: float):
    best, best_val = None, -1.0
    for o, bu, su in pareto:
        val = max(0.0, bu - buyer_rv) * max(0.0, su - seller_rv)
        if val > best_val:
            best_val, best = val, (o, bu, su)
    return best


def _pareto_dist(bu, su, pareto) -> float:
    if not pareto:
        return float("nan")
    return min(
        math.sqrt((bu - p[1]) ** 2 + (su - p[2]) ** 2) for p in pareto
    ) / math.sqrt(2)


def _nash_dist(bu, su, nash) -> float:
    if nash is None:
        return float("nan")
    return math.sqrt((bu - nash[1]) ** 2 + (su - nash[2]) ** 2) / math.sqrt(2)


def _advantage(u, rv, max_u=1.0):
    return (u - rv) / (max_u - rv) if max_u != rv else 0.0


def _extract_trace(session, buyer_ufun, seller_ufun):
    """Return list of (round, buyer_util, seller_util, proposer) from history."""
    trace = []
    for i, state in enumerate(getattr(session, "history", [])):
        offer = getattr(state, "current_offer", None)
        if offer is None:
            continue
        offer = tuple(offer)
        proposer = getattr(state, "current_proposer", None)
        trace.append(
            (
                i,
                float(buyer_ufun(offer)),
                float(seller_ufun(offer)),
                str(proposer) if proposer else "?",
            )
        )
    return trace


# ─────────────────────────────────────────────────────────────────────────────
# Run negotiations
# ─────────────────────────────────────────────────────────────────────────────


def _short(type_name: str) -> str:
    return type_name.replace("TBNegotiator", "TB").replace("Negotiator", "")


def _make_agent(type_name: str, role: str):
    """Create agent with a label that encodes both type and role.
    E.g. 'Adaptive-buyer', 'BoulwareTB-seller'.
    This survives NegMAS uniqueness suffixes like (0)/(1).
    """
    label = f"{_short(type_name)}-{role}"
    if type_name == "AdaptivePrONeg":
        return AdaptivePrONeg(name=label)
    return BoulwareTBNegotiator(name=label)


def run_negotiations(
    scenario_fn, scenario_name: str, n_runs: int = 20, n_steps: int = 30
) -> pd.DataFrame:
    # Pre-compute outcome space once
    s0, sf0, bf0 = scenario_fn(n_steps=n_steps)
    outcomes = _get_outcomes(s0)
    brv, srv = float(bf0.reserved_value or 0.0), float(sf0.reserved_value or 0.0)
    pareto0 = _pareto_frontier(outcomes, bf0, sf0) if outcomes else []
    nash0 = _nash_point(pareto0, brv, srv)

    rows = []
    for run in range(n_runs):
        for buyer_type, seller_type in COMBOS:
            session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
            session.add(_make_agent(buyer_type, "buyer"), ufun=buyer_ufun)
            session.add(_make_agent(seller_type, "seller"), ufun=seller_ufun)

            t0 = time.perf_counter()
            result = session.run()
            elapsed = time.perf_counter() - t0

            agreement = getattr(result, "agreement", None)
            agreed = agreement is not None
            if agreed:
                agreement = tuple(agreement)

            buyer_rv = float(buyer_ufun.reserved_value or 0.0)
            seller_rv = float(seller_ufun.reserved_value or 0.0)
            bu = float(buyer_ufun(agreement)) if agreed else buyer_rv
            su = float(seller_ufun(agreement)) if agreed else seller_rv

            rows.append(
                {
                    "run": run,
                    "scenario": scenario_name,
                    "combo": f"{buyer_type} vs {seller_type}",
                    "buyer_type": buyer_type,
                    "seller_type": seller_type,
                    "agreed": agreed,
                    "buyer_utility": bu,
                    "seller_utility": su,
                    "social_welfare": bu + su,
                    "buyer_advantage": _advantage(bu, buyer_rv),
                    "seller_advantage": _advantage(su, seller_rv),
                    "pareto_dist": (
                        _pareto_dist(bu, su, pareto0) if agreed else float("nan")
                    ),
                    "nash_dist": _nash_dist(bu, su, nash0) if agreed else float("nan"),
                    "n_steps": (
                        len(session.history) if getattr(session, "history", None) else 0
                    ),
                    "time_s": elapsed,
                }
            )
    return pd.DataFrame(rows)


def run_trace_negotiations(scenario_fn, n_steps: int = 30) -> dict:
    """Run one negotiation per combo and return history traces."""
    s0, sf0, bf0 = scenario_fn(n_steps=n_steps)
    outcomes = _get_outcomes(s0)
    brv, srv = float(bf0.reserved_value or 0.0), float(sf0.reserved_value or 0.0)
    pareto0 = _pareto_frontier(outcomes, bf0, sf0) if outcomes else []
    nash0 = _nash_point(pareto0, brv, srv)

    traces = {}
    for buyer_type, seller_type in COMBOS:
        session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
        session.add(_make_agent(buyer_type, "buyer"), ufun=buyer_ufun)
        session.add(_make_agent(seller_type, "seller"), ufun=seller_ufun)
        session.run()
        combo = f"{buyer_type} vs {seller_type}"
        traces[combo] = {
            "trace": _extract_trace(session, buyer_ufun, seller_ufun),
            "pareto": pareto0,
            "nash": nash0,
            "buyer_rv": brv,
            "seller_rv": srv,
            "buyer_ufun": buyer_ufun,
            "seller_ufun": seller_ufun,
        }
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation report
# ─────────────────────────────────────────────────────────────────────────────


def print_evaluation(df: pd.DataFrame, scenario_name: str):
    agreed = df[df["agreed"]]

    print(f"\n[bold cyan]{'═'*55}[/bold cyan]")
    print(f"[bold cyan]  {scenario_name}[/bold cyan]")
    print(f"[bold cyan]{'═'*55}[/bold cyan]")

    print(
        "\n[bold yellow]── Agreement Rates ──────────────────────────────────[/bold yellow]"
    )
    agg = (
        df.groupby("combo")
        .agg(total=("agreed", "count"), agreements=("agreed", "sum"))
        .assign(agreement_rate=lambda x: x["agreements"] / x["total"] * 100)
    )
    agg["agreement_rate"] = agg["agreement_rate"].map("{:.1f}%".format)
    print(agg[["total", "agreements", "agreement_rate"]].to_string())

    print(
        "\n[bold yellow]── Utility & Welfare (agreements only) ─────────────[/bold yellow]"
    )
    uw = (
        agreed.groupby("combo")[
            ["buyer_utility", "seller_utility", "social_welfare", "buyer_advantage"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    uw.columns = ["_".join(c) for c in uw.columns]
    print(
        uw[
            [
                "buyer_utility_mean",
                "seller_utility_mean",
                "social_welfare_mean",
                "buyer_advantage_mean",
            ]
        ]
        .sort_values("buyer_advantage_mean", ascending=False)
        .to_string()
    )

    print(
        "\n[bold yellow]── Distance Metrics (lower = better) ───────────────[/bold yellow]"
    )
    dist = (
        agreed.groupby("combo")[["pareto_dist", "nash_dist"]]
        .agg(["mean", "std"])
        .round(4)
    )
    dist.columns = ["_".join(c) for c in dist.columns]
    print(dist.sort_values("nash_dist_mean").to_string())

    print(
        "\n[bold yellow]── Rank Summary (Sec 2.4 criteria) ─────────────────[/bold yellow]"
    )
    rank = (
        pd.DataFrame(
            {
                "agree_%": df.groupby("combo")["agreed"].mean() * 100,
                "advantage": agreed.groupby("combo")["buyer_advantage"].mean(),
                "social_welfare": agreed.groupby("combo")["social_welfare"].mean(),
                "nash_dist": agreed.groupby("combo")["nash_dist"].mean(),
                "pareto_dist": agreed.groupby("combo")["pareto_dist"].mean(),
            }
        )
        .sort_values("advantage", ascending=False)
        .round(4)
    )
    rank["agree_%"] = rank["agree_%"].map("{:.1f}%".format)
    print(rank.to_string())
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Utility Space
# Shows where agreements land relative to the Pareto frontier and Nash point.
# Directly answers "how close to Pareto?" and "how close to Nash?" (Sec 2.4).
# ─────────────────────────────────────────────────────────────────────────────


def plot_utility_space(
    scenario_fn, df: pd.DataFrame, scenario_name: str, n_steps: int = 30
):
    session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
    outcomes = _get_outcomes(session)
    if not outcomes:
        return

    brv = float(buyer_ufun.reserved_value or 0.0)
    srv = float(seller_ufun.reserved_value or 0.0)
    all_bu = [float(buyer_ufun(o)) for o in outcomes]
    all_su = [float(seller_ufun(o)) for o in outcomes]
    pareto = _pareto_frontier(outcomes, buyer_ufun, seller_ufun)
    nash = _nash_point(pareto, brv, srv)

    fig, ax = plt.subplots(figsize=(8, 7))

    # All outcomes (background)
    ax.scatter(
        all_bu,
        all_su,
        s=10,
        alpha=0.12,
        color="#bdbdbd",
        zorder=1,
        label="All outcomes",
    )

    # ZOPA region (shaded)
    ax.fill_betweenx(
        [srv, 1.05], brv, 1.05, alpha=0.06, color="green", label="ZOPA region"
    )

    # Pareto frontier
    if pareto:
        p_sorted = sorted(pareto, key=lambda x: x[1])
        px, py = [p[1] for p in p_sorted], [p[2] for p in p_sorted]
        ax.plot(
            px, py, color="steelblue", linewidth=2.0, zorder=3, label="Pareto frontier"
        )
        ax.scatter(px, py, s=35, color="steelblue", zorder=4)

    # Nash point
    if nash:
        ax.scatter(
            [nash[1]],
            [nash[2]],
            s=280,
            marker="D",
            color="gold",
            edgecolors="black",
            linewidths=1.5,
            zorder=6,
            label="Nash point",
        )

    # Agreement points per combo
    agreed = df[df["agreed"]]
    for combo in df["combo"].unique():
        grp = agreed[agreed["combo"] == combo]
        if grp.empty:
            continue
        ax.scatter(
            grp["buyer_utility"],
            grp["seller_utility"],
            s=60,
            alpha=0.70,
            color=COLORS.get(combo, "#888"),
            label=f"{COMBO_SHORT[combo]}",
            zorder=5,
        )

    # Reservation lines
    ax.axvline(
        brv, color="#e53935", linestyle="--", linewidth=1.5, label=f"Buyer rv={brv:.2f}"
    )
    ax.axhline(
        srv, color="#b71c1c", linestyle=":", linewidth=1.5, label=f"Seller rv={srv:.2f}"
    )

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Buyer Utility", fontsize=12)
    ax.set_ylabel("Seller Utility", fontsize=12)
    ax.set_title(f"Utility Space — {scenario_name}", fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Concession Paths through Utility Space
# Shows how each combo's offers trace through the utility space over time.
# Earlier offers are lighter; later offers are darker. Arrows show direction.
# Reveals the stark difference in concession behaviour.
# ─────────────────────────────────────────────────────────────────────────────


def plot_concession_paths(scenario_fn, scenario_name: str, n_steps: int = 30):
    traces = run_trace_negotiations(scenario_fn, n_steps=n_steps)
    combos = list(traces.keys())
    n = len(combos)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # Use the outcome space from the first trace entry
    sample = next(iter(traces.values()))
    pareto = sample["pareto"]
    nash = sample["nash"]
    brv = sample["buyer_rv"]
    srv = sample["seller_rv"]

    for ax, combo in zip(axes, combos):
        info = traces[combo]
        trace = info["trace"]

        if pareto:
            p_sorted = sorted(pareto, key=lambda x: x[1])
            ax.plot(
                [p[1] for p in p_sorted],
                [p[2] for p in p_sorted],
                color="steelblue",
                linewidth=1.5,
                alpha=0.6,
                zorder=2,
                label="Pareto frontier",
            )

        if nash:
            ax.scatter(
                [nash[1]],
                [nash[2]],
                s=180,
                marker="D",
                color="gold",
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
                label="Nash",
            )

        ax.axvline(brv, color="#e53935", linestyle="--", linewidth=1.2)
        ax.axhline(srv, color="#b71c1c", linestyle=":", linewidth=1.2)

        if trace:
            bu_vals = [t[1] for t in trace]
            su_vals = [t[2] for t in trace]
            n_pts = len(trace)

            # Colour gradient: light → dark over time
            cmap = plt.cm.get_cmap("plasma")
            for i in range(n_pts - 1):
                frac = i / max(n_pts - 1, 1)
                c = cmap(frac)
                ax.annotate(
                    "",
                    xy=(bu_vals[i + 1], su_vals[i + 1]),
                    xytext=(bu_vals[i], su_vals[i]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.3),
                )
            # Scatter with time-based colour
            ax.scatter(
                bu_vals,
                su_vals,
                c=range(n_pts),
                cmap="plasma",
                s=35,
                zorder=4,
                vmin=0,
                vmax=max(n_pts - 1, 1),
            )

            # Mark start and end
            ax.scatter(
                [bu_vals[0]],
                [su_vals[0]],
                s=120,
                marker="o",
                color="lime",
                edgecolors="black",
                zorder=6,
                label="Start",
            )
            ax.scatter(
                [bu_vals[-1]],
                [su_vals[-1]],
                s=120,
                marker="*",
                color="white",
                edgecolors="black",
                zorder=6,
                label="End",
            )

        ax.set_xlim(-0.02, 1.08)
        ax.set_ylim(-0.02, 1.08)
        ax.set_title(COMBO_SHORT[combo], fontweight="bold", fontsize=10)
        ax.set_xlabel("Buyer Utility")
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.set_ylabel("Seller Utility")
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        f"Concession Paths — {scenario_name}\n"
        "(light = early offers, dark = late offers)",
        fontweight="bold",
        fontsize=12,
    )
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Self-interest vs Joint Quality Scatter
# Each dot = one negotiation run, x = buyer advantage (self-interest),
# y = social welfare (joint quality).
# Reveals the trade-off: agents at the top-right dominate.
# ─────────────────────────────────────────────────────────────────────────────


def plot_self_vs_joint(all_dfs: list[pd.DataFrame], scenario_names: list[str]):
    n = len(scenario_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (sname, df) in zip(axes, zip(scenario_names, all_dfs)):
        agreed = df[df["agreed"]]
        for combo in df["combo"].unique():
            grp = agreed[agreed["combo"] == combo]
            if grp.empty:
                continue
            color = COLORS.get(combo, "#888")
            # Individual run points
            ax.scatter(
                grp["buyer_advantage"],
                grp["social_welfare"],
                s=30,
                alpha=0.45,
                color=color,
                zorder=3,
            )
            # Mean marker
            mx, my = grp["buyer_advantage"].mean(), grp["social_welfare"].mean()
            ax.scatter(
                [mx],
                [my],
                s=200,
                marker="D",
                color=color,
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
                label=COMBO_SHORT[combo],
            )
            # 95% CI ellipse approximation (±1.96 std)
            ex = grp["buyer_advantage"].std() * 1.96
            ey = grp["social_welfare"].std() * 1.96
            ellipse = plt.matplotlib.patches.Ellipse(
                (mx, my),
                width=ex * 2,
                height=ey * 2,
                color=color,
                alpha=0.12,
                zorder=2,
            )
            ax.add_patch(ellipse)

        ax.set_xlabel("Buyer Advantage  (higher = better for buyer)", fontsize=10)
        ax.set_ylabel("Social Welfare   (higher = better for both)", fontsize=10)
        ax.set_title(sname, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

        # Quadrant annotations
        ax.text(
            0.98,
            0.02,
            "High self\nLow joint",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
        )
        ax.text(
            0.98,
            0.98,
            "High self\nHigh joint",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="green",
        )

    fig.suptitle(
        "Self-interest vs Joint Quality Trade-off\n"
        "(diamonds = mean per combo; ellipses = 95% CI)",
        fontweight="bold",
        fontsize=12,
    )
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Radar (Spider) Chart
# Shows 5 metrics simultaneously for each combo.
# All axes: higher = better (distances converted to closeness).
# Metrics address all three assignment criteria (Sec 2.4) plus self-performance.
# ─────────────────────────────────────────────────────────────────────────────


def plot_radar_summary(all_dfs: list[pd.DataFrame], scenario_names: list[str]):
    METRICS = [
        ("agree_%", "Agreement\nRate"),
        ("advantage", "Buyer\nAdvantage"),
        ("social_welfare", "Social\nWelfare"),
        ("nash_closeness", "Nash\nCloseness"),
        ("pareto_closeness", "Pareto\nCloseness"),
    ]
    n_metrics = len(METRICS)
    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]

    n_scen = len(scenario_names)
    fig, axes = plt.subplots(
        1, n_scen, figsize=(6 * n_scen, 5), subplot_kw=dict(polar=True)
    )
    if n_scen == 1:
        axes = [axes]

    for ax, (sname, df) in zip(axes, zip(scenario_names, all_dfs)):
        agreed = df[df["agreed"]]

        # Build metric table for each combo
        combo_metrics = {}
        for combo in df["combo"].unique():
            grp = agreed[agreed["combo"] == combo]
            if grp.empty:
                continue
            combo_metrics[combo] = {
                "agree_%": df[df["combo"] == combo]["agreed"].mean(),
                "advantage": grp["buyer_advantage"].mean(),
                "social_welfare": grp["social_welfare"].mean()
                / 2,  # normalise to [0,1]
                "nash_closeness": 1 - grp["nash_dist"].mean(),
                "pareto_closeness": 1 - grp["pareto_dist"].mean(),
            }

        # Min-max normalise across combos so each axis spans [0, 1]
        all_keys = [m[0] for m in METRICS]
        mins = {k: min(v[k] for v in combo_metrics.values()) for k in all_keys}
        maxs = {k: max(v[k] for v in combo_metrics.values()) for k in all_keys}

        for combo, vals in combo_metrics.items():
            normed = [
                (vals[k] - mins[k]) / (maxs[k] - mins[k] + 1e-9) for k in all_keys
            ]
            normed += normed[:1]
            color = COLORS.get(combo, "#888")
            ax.plot(angles, normed, color=color, linewidth=2, label=COMBO_SHORT[combo])
            ax.fill(angles, normed, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[1] for m in METRICS], fontsize=9)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6, color="gray")
        ax.set_ylim(0, 1)
        ax.set_title(sname, fontweight="bold", fontsize=11, pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    fig.suptitle(
        "Performance Radar — All Metrics\n"
        "(axes normalised within scenario; outer = better)",
        fontweight="bold",
        fontsize=12,
    )
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Built-in NegMAS session plot
# Runs one negotiation per combo and calls session.plot(), which shows the
# NegMAS standard visualisation: offer trace, Pareto distance, agreement point,
# Nash/Kalai markers.  One figure per combo per scenario.
# ─────────────────────────────────────────────────────────────────────────────


def plot_builtin_negmas(scenario_fn, scenario_name: str, n_steps: int = 30):
    # Only show the two cross-agent combos — symmetric ones add no new insight
    cross_combos = [(b, s) for b, s in COMBOS if b != s]
    for buyer_type, seller_type in cross_combos:
        session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
        session.add(_make_agent(buyer_type, "buyer"), ufun=buyer_ufun)
        session.add(_make_agent(seller_type, "seller"), ufun=seller_ufun)
        session.run()

        combo_label = (
            f"{_short(buyer_type)}-buyer  vs  "
            f"{_short(seller_type)}-seller — {scenario_name}"
        )
        try:
            fig = session.plot(show_reserved=True)
        except TypeError:
            fig = session.plot()

        # session.plot() creates and returns its own figure; retitle it
        if fig is not None:
            fig.suptitle(
                f"NegMAS Built-in Plot\n{combo_label}", fontweight="bold", fontsize=11
            )
        else:
            # Some NegMAS versions return None and use plt.gcf()
            plt.suptitle(
                f"NegMAS Built-in Plot\n{combo_label}", fontweight="bold", fontsize=11
            )
        plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    N_RUNS = 20
    N_STEPS = 30

    print(
        "[bold green]╔══════════════════════════════════════════════════╗[/bold green]"
    )
    print("[bold green]║  AdaptivePrONeg vs BoulwareTBNegotiator      ║[/bold green]")
    print(
        "[bold green]╚══════════════════════════════════════════════════╝[/bold green]"
    )
    print(f"Runs per combination: {N_RUNS}  |  Steps per negotiation: {N_STEPS}")

    all_dfs = []
    scenario_names = list(SCENARIOS.keys())

    for sname, fn in SCENARIOS.items():
        print(f"\n[cyan]Running scenario: {sname}...[/cyan]")
        df = run_negotiations(fn, sname, n_runs=N_RUNS, n_steps=N_STEPS)
        all_dfs.append(df)
        print_evaluation(df, sname)

    print("\n[bold green]Generating plots...[/bold green]")

    # Plot 1 – Utility Space (one per scenario)
    for sname, fn, df in zip(scenario_names, SCENARIOS.values(), all_dfs):
        plot_utility_space(fn, df, sname, n_steps=N_STEPS)

    # Plot 2 – Concession Paths (one per scenario)
    for sname, fn in zip(scenario_names, SCENARIOS.values()):
        print(f"  [dim]Running trace negotiations for: {sname}[/dim]")
        plot_concession_paths(fn, sname, n_steps=N_STEPS)

    # Plot 3 – Self vs Joint trade-off
    plot_self_vs_joint(all_dfs, scenario_names)

    # Plot 4 – Radar summary
    plot_radar_summary(all_dfs, scenario_names)

    # Plot 5 – Built-in NegMAS session plot (one per combo per scenario)
    for sname, fn in zip(scenario_names, SCENARIOS.values()):
        print(f"  [dim]Running NegMAS built-in plots for: {sname}[/dim]")
        plot_builtin_negmas(fn, sname, n_steps=N_STEPS)

    plt.show()


if __name__ == "__main__":
    main()
