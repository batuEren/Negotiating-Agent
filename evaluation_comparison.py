"""
evaluation_comparison.py
========================
Comprehensive head-to-head evaluation of MicroNegotiator vs AdaptiveNegotiator.

Metrics covered
---------------
- Agreement rate (by role and by matchup)
- Utility: mean, std, min, max (buyer & seller, agreed rounds only)
- Social welfare and fairness (|u_buyer - u_seller|)
- Nash optimality (distance from Nash point)
- Pareto optimality (fraction of agreements on the Pareto frontier)
- Negotiation length (steps to agreement or deadline)
- Role advantage (does playing buyer vs seller matter?)

Scenarios
---------
1. Single-issue price negotiation (simple, reproducible)
2. Multi-issue (price × quantity × delivery_time) negotiation

Run
---
    python evaluation_comparison.py
"""

from __future__ import annotations

import math
import warnings
from itertools import product as iproduct

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from rich import print
from rich.table import Table
from rich.console import Console

from negmas import make_issue, SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun

from microNegotiator import MicroNegotiator
from adaptive_agent import AdaptiveNegotiator

warnings.filterwarnings("ignore")
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Scenario factories
# ─────────────────────────────────────────────────────────────────────────────


def make_single_issue_scenario(n_steps: int = 30):
    """Single price issue, overlapping ZOPA (prices 6-8 are mutually acceptable)."""
    issues = [make_issue(name="price", values=13)]  # 0..12
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    seller_utility = LUFun(
        values={"price": IdentityFun()},
        outcome_space=session.outcome_space,
        reserved_value=6 / 12.0,
    ).scale_max(1.0)

    buyer_utility = LUFun(
        values={"price": AffineFun(-1, bias=12.0)},
        outcome_space=session.outcome_space,
        reserved_value=(12.0 - 8) / 12.0,
    ).scale_max(1.0)

    return session, buyer_utility, seller_utility


def make_multi_issue_scenario(n_steps: int = 30):
    """Three-issue negotiation (price × quantity × delivery_time)."""
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=(1, 11)),
        make_issue(name="delivery_time", values=10),
    ]
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    seller_utility = LUFun(
        values={
            "price": IdentityFun(),
            "quantity": LinearFun(0.2),
            "delivery_time": AffineFun(-1, bias=9.0),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.3,
    ).scale_max(1.0)

    buyer_utility = LUFun(
        values={
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.2,
    ).scale_max(1.0)

    return session, buyer_utility, seller_utility


# ─────────────────────────────────────────────────────────────────────────────
# Pareto / Nash helpers
# ─────────────────────────────────────────────────────────────────────────────


def _all_utilities(session, buyer_ufun, seller_ufun):
    """Return arrays of (buyer_u, seller_u) for every outcome."""
    outcomes = list(session.outcome_space.enumerate_or_sample())
    bus = np.array([float(buyer_ufun(o)) for o in outcomes])
    sus = np.array([float(seller_ufun(o)) for o in outcomes])
    return bus, sus, outcomes


def _pareto_front(bus, sus):
    """Boolean mask: True if outcome is Pareto-efficient (max-max)."""
    n = len(bus)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = (
            (bus >= bus[i]) & (sus >= sus[i]) & ((bus > bus[i]) | (sus > sus[i]))
        )
        if dominated.any():
            mask[i] = False
    return mask


def _nash_point(bus, sus, buyer_rv, seller_rv):
    """Index of the Nash bargaining solution."""
    gains = (bus - buyer_rv).clip(min=0) * (sus - seller_rv).clip(min=0)
    return int(np.argmax(gains))


def _is_pareto(offer, session, buyer_ufun, seller_ufun):
    bus, sus, outcomes = _all_utilities(session, buyer_ufun, seller_ufun)
    mask = _pareto_front(bus, sus)
    pareto_set = {tuple(o) for o, m in zip(outcomes, mask) if m}
    return tuple(offer) in pareto_set


def _nash_distance(offer, session, buyer_ufun, seller_ufun):
    bus, sus, outcomes = _all_utilities(session, buyer_ufun, seller_ufun)
    buyer_rv = float(buyer_ufun.reserved_value or 0.0)
    seller_rv = float(seller_ufun.reserved_value or 0.0)
    ni = _nash_point(bus, sus, buyer_rv, seller_rv)
    nash_bu, nash_su = bus[ni], sus[ni]
    bu = float(buyer_ufun(offer))
    su = float(seller_ufun(offer))
    return math.sqrt((bu - nash_bu) ** 2 + (su - nash_su) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Run one negotiation, return a result dict
# ─────────────────────────────────────────────────────────────────────────────

AGENT_CLASSES = {
    "Micro": MicroNegotiator,
    "Adaptive": AdaptiveNegotiator,
}

MATCHUPS = [
    ("Micro", "Micro"),
    ("Micro", "Adaptive"),
    ("Adaptive", "Micro"),
    ("Adaptive", "Adaptive"),
]

SCENARIOS = {
    "Single-issue": make_single_issue_scenario,
    "Multi-issue": make_multi_issue_scenario,
}


def run_one(buyer_name: str, seller_name: str, scenario_fn, n_steps: int = 30) -> dict:
    session, buyer_ufun, seller_ufun = scenario_fn(n_steps)

    buyer_agent = AGENT_CLASSES[buyer_name](name=f"buyer_{buyer_name}")
    seller_agent = AGENT_CLASSES[seller_name](name=f"seller_{seller_name}")

    session.add(buyer_agent, ufun=buyer_ufun)
    session.add(seller_agent, ufun=seller_ufun)

    result = session.run()
    agreement = getattr(result, "agreement", None)
    if agreement is not None:
        agreement = tuple(agreement)

    agreed = agreement is not None
    buyer_rv = float(buyer_ufun.reserved_value or 0.0)
    seller_rv = float(seller_ufun.reserved_value or 0.0)

    if agreed:
        bu = float(buyer_ufun(agreement))
        su = float(seller_ufun(agreement))
        welfare = bu + su
        fairness = abs(bu - su)
        nash_dist = _nash_distance(agreement, session, buyer_ufun, seller_ufun)
        on_pareto = _is_pareto(agreement, session, buyer_ufun, seller_ufun)
        steps = len(session.history)
    else:
        bu = buyer_rv
        su = seller_rv
        welfare = bu + su
        fairness = abs(bu - su)
        nash_dist = float("nan")
        on_pareto = False
        steps = n_steps

    return {
        "buyer": buyer_name,
        "seller": seller_name,
        "matchup": f"{buyer_name} (B) vs {seller_name} (S)",
        "agreed": agreed,
        "buyer_utility": bu,
        "seller_utility": su,
        "welfare": welfare,
        "fairness": fairness,
        "nash_distance": nash_dist,
        "on_pareto": on_pareto,
        "steps": steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run full tournament
# ─────────────────────────────────────────────────────────────────────────────


def run_tournament(n_runs: int = 20, n_steps: int = 30) -> dict[str, pd.DataFrame]:
    dfs = {}
    for scenario_name, scenario_fn in SCENARIOS.items():
        rows = []
        for run_id in range(n_runs):
            for buyer_name, seller_name in MATCHUPS:
                row = run_one(buyer_name, seller_name, scenario_fn, n_steps)
                row["run_id"] = run_id
                row["scenario"] = scenario_name
                rows.append(row)
        dfs[scenario_name] = pd.DataFrame(rows)
    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# Rich console report
# ─────────────────────────────────────────────────────────────────────────────


def _pct(v):
    return f"{v * 100:.1f}%"


def _f(v, d=3):
    return f"{v:.{d}f}" if not math.isnan(v) else "—"


def print_report(dfs: dict[str, pd.DataFrame]):
    console.rule("[bold cyan]MICRO vs ADAPTIVE — COMPREHENSIVE EVALUATION[/bold cyan]")

    for scenario_name, df in dfs.items():
        console.rule(f"[bold yellow]Scenario: {scenario_name}[/bold yellow]")
        agreed = df[df["agreed"]].copy()

        # ── Agreement rate ──────────────────────────────────────────────
        t = Table(title="Agreement Rate by Matchup", show_lines=True)
        t.add_column("Matchup", style="bold")
        t.add_column("Trials", justify="right")
        t.add_column("Agreements", justify="right")
        t.add_column("Rate", justify="right", style="green")

        for matchup in df["matchup"].unique():
            sub = df[df["matchup"] == matchup]
            n = len(sub)
            a = sub["agreed"].sum()
            t.add_row(matchup, str(n), str(a), _pct(a / n))
        console.print(t)

        if agreed.empty:
            console.print(
                "[red]No agreements reached — skipping utility/optimality tables.[/red]"
            )
            continue

        # ── Utility & welfare ───────────────────────────────────────────
        t2 = Table(title="Utility & Welfare (agreed rounds only)", show_lines=True)
        for col in [
            "Matchup",
            "Buyer Util (mean±std)",
            "Seller Util (mean±std)",
            "Welfare",
            "Fairness (|Δu|)",
        ]:
            t2.add_column(col)

        for matchup in agreed["matchup"].unique():
            sub = agreed[agreed["matchup"] == matchup]
            bu_m, bu_s = sub["buyer_utility"].mean(), sub["buyer_utility"].std()
            su_m, su_s = sub["seller_utility"].mean(), sub["seller_utility"].std()
            wm = sub["welfare"].mean()
            fm = sub["fairness"].mean()
            t2.add_row(
                matchup,
                f"{bu_m:.3f} ± {bu_s:.3f}",
                f"{su_m:.3f} ± {su_s:.3f}",
                f"{wm:.3f}",
                f"{fm:.3f}",
            )
        console.print(t2)

        # ── Optimality ──────────────────────────────────────────────────
        t3 = Table(title="Optimality (agreed rounds only)", show_lines=True)
        for col in [
            "Matchup",
            "Nash Distance (mean)",
            "On Pareto (%)",
            "Steps to Agreement",
        ]:
            t3.add_column(col)

        for matchup in agreed["matchup"].unique():
            sub = agreed[agreed["matchup"] == matchup]
            all_sub = df[df["matchup"] == matchup]
            nd = sub["nash_distance"].mean()
            pareto_rate = sub["on_pareto"].mean()
            steps_m = all_sub["steps"].mean()
            t3.add_row(
                matchup,
                _f(nd),
                _pct(pareto_rate),
                f"{steps_m:.1f}",
            )
        console.print(t3)

        # ── Role advantage ──────────────────────────────────────────────
        t4 = Table(title="Role Advantage: Micro vs Adaptive", show_lines=True)
        t4.add_column("Agent", style="bold")
        t4.add_column("Role", style="bold")
        t4.add_column("Mean Utility (agreed)")
        t4.add_column("Agreement Rate")

        for agent in ("Micro", "Adaptive"):
            for role in ("buyer", "seller"):
                sub_all = df[df[role] == agent]
                sub_agr = agreed[agreed[role] == agent]
                mean_u = (
                    sub_agr[f"{role}_utility"].mean()
                    if not sub_agr.empty
                    else float("nan")
                )
                rate = sub_all["agreed"].mean()
                t4.add_row(agent, role.capitalize(), _f(mean_u), _pct(rate))
        console.print(t4)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Micro (B) vs Micro (S)": "#4C72B0",
    "Micro (B) vs Adaptive (S)": "#DD8452",
    "Adaptive (B) vs Micro (S)": "#55A868",
    "Adaptive (B) vs Adaptive (S)": "#C44E52",
}


def _color(matchup):
    return COLORS.get(matchup, "#888888")


def plot_all(dfs: dict[str, pd.DataFrame]):
    n_scenarios = len(dfs)
    fig = plt.figure(figsize=(20, 6 * n_scenarios * 2))
    outer = gridspec.GridSpec(n_scenarios, 1, figure=fig, hspace=0.55)

    for s_idx, (scenario_name, df) in enumerate(dfs.items()):
        agreed = df[df["agreed"]].copy()
        matchups = df["matchup"].unique()
        colors = [_color(m) for m in matchups]

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 4, subplot_spec=outer[s_idx], hspace=0.5, wspace=0.4
        )

        # ── 1. Agreement rate bar ───────────────────────────────────────
        ax1 = fig.add_subplot(inner[0, 0])
        rates = [df[df["matchup"] == m]["agreed"].mean() * 100 for m in matchups]
        bars = ax1.bar(range(len(matchups)), rates, color=colors)
        ax1.set_xticks(range(len(matchups)))
        ax1.set_xticklabels([m.replace(" vs ", "\nvs\n") for m in matchups], fontsize=7)
        ax1.set_ylabel("Agreement Rate (%)")
        ax1.set_title(f"[{scenario_name}]\nAgreement Rate")
        ax1.set_ylim(0, 110)
        for bar, rate in zip(bars, rates):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{rate:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # ── 2. Mean buyer utility (agreed) ─────────────────────────────
        ax2 = fig.add_subplot(inner[0, 1])
        if not agreed.empty:
            bu_means = [
                agreed[agreed["matchup"] == m]["buyer_utility"].mean() for m in matchups
            ]
            bu_stds = [
                agreed[agreed["matchup"] == m]["buyer_utility"].std() for m in matchups
            ]
            ax2.bar(
                range(len(matchups)), bu_means, yerr=bu_stds, color=colors, capsize=4
            )
        ax2.set_xticks(range(len(matchups)))
        ax2.set_xticklabels([m.replace(" vs ", "\nvs\n") for m in matchups], fontsize=7)
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Mean Buyer Utility\n(agreed)")
        ax2.set_ylabel("Utility")

        # ── 3. Mean seller utility (agreed) ────────────────────────────
        ax3 = fig.add_subplot(inner[0, 2])
        if not agreed.empty:
            su_means = [
                agreed[agreed["matchup"] == m]["seller_utility"].mean()
                for m in matchups
            ]
            su_stds = [
                agreed[agreed["matchup"] == m]["seller_utility"].std() for m in matchups
            ]
            ax3.bar(
                range(len(matchups)), su_means, yerr=su_stds, color=colors, capsize=4
            )
        ax3.set_xticks(range(len(matchups)))
        ax3.set_xticklabels([m.replace(" vs ", "\nvs\n") for m in matchups], fontsize=7)
        ax3.set_ylim(0, 1.1)
        ax3.set_title("Mean Seller Utility\n(agreed)")
        ax3.set_ylabel("Utility")

        # ── 4. Social welfare (agreed) ──────────────────────────────────
        ax4 = fig.add_subplot(inner[0, 3])
        if not agreed.empty:
            wm = [agreed[agreed["matchup"] == m]["welfare"].mean() for m in matchups]
            ax4.bar(range(len(matchups)), wm, color=colors)
        ax4.set_xticks(range(len(matchups)))
        ax4.set_xticklabels([m.replace(" vs ", "\nvs\n") for m in matchups], fontsize=7)
        ax4.set_ylim(0, 2.1)
        ax4.set_title("Social Welfare\n(agreed)")
        ax4.set_ylabel("Welfare")
        ax4.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="neutral")

        # ── 5. Utility scatter in utility space (agreed) ────────────────
        ax5 = fig.add_subplot(inner[1, 0:2])
        if not agreed.empty:
            for m, c in zip(matchups, colors):
                sub = agreed[agreed["matchup"] == m]
                ax5.scatter(
                    sub["buyer_utility"],
                    sub["seller_utility"],
                    label=m,
                    color=c,
                    alpha=0.6,
                    s=40,
                )
        ax5.set_xlabel("Buyer Utility")
        ax5.set_ylabel("Seller Utility")
        ax5.set_title("Agreements in Utility Space")
        ax5.set_xlim(0, 1.1)
        ax5.set_ylim(0, 1.1)
        ax5.legend(fontsize=7, loc="upper right")
        ax5.grid(True, alpha=0.3)

        # ── 6. Negotiation length box-plot ──────────────────────────────
        ax6 = fig.add_subplot(inner[1, 2])
        data_for_box = [df[df["matchup"] == m]["steps"].values for m in matchups]
        bp = ax6.boxplot(
            data_for_box,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax6.set_xticks(range(1, len(matchups) + 1))
        ax6.set_xticklabels([m.replace(" vs ", "\nvs\n") for m in matchups], fontsize=7)
        ax6.set_title("Steps to Agreement\n(or deadline)")
        ax6.set_ylabel("Steps")

        # ── 7. Pareto & Nash optimality ─────────────────────────────────
        ax7 = fig.add_subplot(inner[1, 3])
        if not agreed.empty:
            pareto_rates = [
                agreed[agreed["matchup"] == m]["on_pareto"].mean() * 100
                for m in matchups
            ]
            x = np.arange(len(matchups))
            ax7.bar(x, pareto_rates, color=colors, alpha=0.85)
            ax7.set_xticks(x)
            ax7.set_xticklabels(
                [m.replace(" vs ", "\nvs\n") for m in matchups], fontsize=7
            )
            ax7.set_ylim(0, 110)
            ax7.set_title("Pareto Optimality\n(% agreements on frontier)")
            ax7.set_ylabel("% on Pareto frontier")

    fig.suptitle(
        "MicroNegotiator vs AdaptiveNegotiator — Full Comparison", fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.savefig("comparison_results.png", dpi=150, bbox_inches="tight")
    print("\n[green]Saved plot → comparison_results.png[/green]")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent summary (agent-level, not matchup-level)
# ─────────────────────────────────────────────────────────────────────────────


def print_agent_summary(dfs: dict[str, pd.DataFrame]):
    console.rule(
        "[bold cyan]AGENT-LEVEL SUMMARY (pooled across roles & scenarios)[/bold cyan]"
    )

    all_rows = pd.concat(dfs.values(), ignore_index=True)
    agreed_all = all_rows[all_rows["agreed"]].copy()

    # For each agent: collect utilities when playing buyer OR seller
    for agent in ("Micro", "Adaptive"):
        as_buyer = all_rows[all_rows["buyer"] == agent]
        as_seller = all_rows[all_rows["seller"] == agent]
        agreed_buyer = agreed_all[agreed_all["buyer"] == agent]
        agreed_seller = agreed_all[agreed_all["seller"] == agent]

        console.print(f"\n[bold magenta]{agent}Negotiator[/bold magenta]")

        total_games = len(as_buyer) + len(as_seller)
        total_agreed = agreed_all[
            (agreed_all["buyer"] == agent) | (agreed_all["seller"] == agent)
        ].shape[0]

        console.print(f"  Total games played : {total_games}")
        console.print(
            f"  Agreements reached : {total_agreed}  ({100*total_agreed/total_games:.1f}%)"
        )

        if not agreed_buyer.empty:
            console.print(
                f"  As Buyer  — util mean={agreed_buyer['buyer_utility'].mean():.3f}, "
                f"agree%={100*as_buyer['agreed'].mean():.1f}%"
            )
        if not agreed_seller.empty:
            console.print(
                f"  As Seller — util mean={agreed_seller['seller_utility'].mean():.3f}, "
                f"agree%={100*as_seller['agreed'].mean():.1f}%"
            )

        pareto_buyer = (
            agreed_buyer["on_pareto"].mean() if not agreed_buyer.empty else float("nan")
        )
        pareto_seller = (
            agreed_seller["on_pareto"].mean()
            if not agreed_seller.empty
            else float("nan")
        )
        pareto_overall = (
            agreed_buyer["on_pareto"].tolist() + agreed_seller["on_pareto"].tolist()
        )
        pareto_mean = np.mean(pareto_overall) if pareto_overall else float("nan")
        console.print(f"  Pareto rate (overall): {pareto_mean*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    N_RUNS = 5  # negotiations per matchup per scenario
    N_STEPS = 30  # max negotiation steps

    console.print(
        f"[bold]Running tournament: {N_RUNS} runs × {len(MATCHUPS)} matchups × {len(SCENARIOS)} scenarios[/bold]"
    )
    console.print(f"  → {N_RUNS * len(MATCHUPS) * len(SCENARIOS)} total negotiations\n")

    dfs = run_tournament(n_runs=N_RUNS, n_steps=N_STEPS)

    print_report(dfs)
    print_agent_summary(dfs)
    plot_all(dfs)


if __name__ == "__main__":
    main()
