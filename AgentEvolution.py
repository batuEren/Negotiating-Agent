"""
AgentEvolution.py
=================
Compares three evolutionary versions of the adaptive negotiator:

  v1  AdaptiveFrequency  — Frequency Analysis opponent model
  v2  AdaptiveBayesian   — Bayesian opponent model (Bayes rule, Eq. 4-5)
  v3  AdaptivePrONeg   — Bayesian + PrONeg outcome prediction

Two head-to-head comparisons:
  Stage 1: v1 (Freq) vs v2 (Bayesian)   — impact of Bayesian model
  Stage 2: v2 (Bayesian) vs v3 (Full)   — impact of PrONeg prediction

Evaluation criteria (Sec 2.4):
  - Pareto frontier closeness  → pareto_dist  (lower = better)
  - Nash Product closeness     → nash_dist    (lower = better)
  - Social Welfare             → social_welfare (higher = better)

Plots
-----
  1. Utility Space          — agreements vs Pareto / Nash per stage
  2. Concession Paths       — offer traces through utility space
  3. Self vs Joint Scatter  — advantage vs welfare per run
  4. Radar Summary          — 5 metrics for all agents simultaneously
  5. Built-in NegMAS plots  — one per cross-combo per stage
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
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun

from adaptive_frequency import AdaptiveFrequency
from adaptive_bayesian import AdaptiveBayesian
from adaptive_proneg import AdaptivePrONeg


# ─────────────────────────────────────────────────────────────────────────────
# Agent registry
# ─────────────────────────────────────────────────────────────────────────────

AGENTS = {
    "Old Adaptive (Frequency)": AdaptiveFrequency,
    "Old Adaptive (Bayesian)":  AdaptiveBayesian,
    "Final Adaptive Version":   AdaptivePrONeg,
}

# Two evolutionary stages to compare
STAGES = [
    ("Stage 1: Frequency → Bayesian",          "Old Adaptive (Frequency)", "Old Adaptive (Bayesian)"),
    ("Stage 2: Bayesian → Final Adaptive Version", "Old Adaptive (Bayesian)", "Final Adaptive Version"),
]

COLORS = {
    "Old Adaptive (Frequency)": "#FF7043",
    "Old Adaptive (Bayesian)":  "#42A5F5",
    "Final Adaptive Version":   "#66BB6A",
}

# Short labels used only in NegMAS built-in plots to avoid clutter
SHORT = {
    "Old Adaptive (Frequency)": "OldFreq",
    "Old Adaptive (Bayesian)":  "OldBayes",
    "Final Adaptive Version":   "FinalAdaptive",
}

# Combo colors: buyer/seller pair
def _combo_color(buyer: str, seller: str) -> str:
    # blend the two agent colors slightly
    return COLORS.get(buyer, "#888")


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
    """3-issue: price, quantity, delivery_time."""
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
        reserved_value=0.4,
    ).scale_max(1.0)
    buyer_ufun = LUFun(
        values={
            "price":         AffineFun(-1, bias=9.0),
            "quantity":      LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
        reserved_value=0.2,
    ).scale_max(1.0)
    return session, seller_ufun, buyer_ufun


SCENARIOS = {
    "Single-Issue Price":     create_single_issue_scenario,
    "Multi-Issue (3 issues)": create_multi_issue_scenario,
}


# ─────────────────────────────────────────────────────────────────────────────
# Outcome-space helpers  (shared with AdaptiveVSBoulware)
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


def _nash_point(pareto, buyer_rv, seller_rv):
    best, best_val = None, -1.0
    for o, bu, su in pareto:
        val = max(0.0, bu - buyer_rv) * max(0.0, su - seller_rv)
        if val > best_val:
            best_val, best = val, (o, bu, su)
    return best


def _pareto_dist(bu, su, pareto) -> float:
    if not pareto:
        return float("nan")
    return min(math.sqrt((bu - p[1]) ** 2 + (su - p[2]) ** 2) for p in pareto) / math.sqrt(2)


def _nash_dist(bu, su, nash) -> float:
    if nash is None:
        return float("nan")
    return math.sqrt((bu - nash[1]) ** 2 + (su - nash[2]) ** 2) / math.sqrt(2)


def _advantage(u, rv, max_u=1.0):
    return (u - rv) / (max_u - rv) if max_u != rv else 0.0


def _extract_trace(session, buyer_ufun, seller_ufun):
    trace = []
    for i, state in enumerate(getattr(session, "history", [])):
        offer = getattr(state, "current_offer", None)
        if offer is None:
            continue
        offer = tuple(offer)
        proposer = getattr(state, "current_proposer", None)
        trace.append((i, float(buyer_ufun(offer)), float(seller_ufun(offer)),
                      str(proposer) if proposer else "?"))
    return trace


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_agent(key: str, role: str):
    cls   = AGENTS[key]
    label = f"{key}-{role}"
    return cls(name=label)


# ─────────────────────────────────────────────────────────────────────────────
# Run negotiations for one stage (4 combos: AA, AB, BA, BB)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(stage_name: str, agent_a: str, agent_b: str,
              scenario_fn, scenario_name: str,
              n_runs: int = 20, n_steps: int = 30) -> pd.DataFrame:

    s0, sf0, bf0 = scenario_fn(n_steps=n_steps)
    outcomes = _get_outcomes(s0)
    brv, srv = float(bf0.reserved_value or 0.0), float(sf0.reserved_value or 0.0)
    pareto0  = _pareto_frontier(outcomes, bf0, sf0) if outcomes else []
    nash0    = _nash_point(pareto0, brv, srv)

    combos = [
        (agent_a, agent_a),
        (agent_a, agent_b),
        (agent_b, agent_a),
        (agent_b, agent_b),
    ]

    rows = []
    for run in range(n_runs):
        for buyer_key, seller_key in combos:
            session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
            session.add(_make_agent(buyer_key,  "buyer"),  ufun=buyer_ufun)
            session.add(_make_agent(seller_key, "seller"), ufun=seller_ufun)

            t0      = time.perf_counter()
            result  = session.run()
            elapsed = time.perf_counter() - t0

            agreement = getattr(result, "agreement", None)
            agreed    = agreement is not None
            if agreed:
                agreement = tuple(agreement)

            buyer_rv  = float(buyer_ufun.reserved_value  or 0.0)
            seller_rv = float(seller_ufun.reserved_value or 0.0)
            bu = float(buyer_ufun(agreement))  if agreed else buyer_rv
            su = float(seller_ufun(agreement)) if agreed else seller_rv

            rows.append({
                "run":            run,
                "stage":          stage_name,
                "scenario":       scenario_name,
                "combo":          f"{buyer_key} vs {seller_key}",
                "buyer_key":      buyer_key,
                "seller_key":     seller_key,
                "agreed":         agreed,
                "buyer_utility":  bu,
                "seller_utility": su,
                "social_welfare": bu + su,
                "buyer_advantage":  _advantage(bu,  buyer_rv),
                "seller_advantage": _advantage(su, seller_rv),
                "pareto_dist":    _pareto_dist(bu, su, pareto0) if agreed else float("nan"),
                "nash_dist":      _nash_dist(bu,   su, nash0)   if agreed else float("nan"),
                "n_steps":        len(session.history) if getattr(session, "history", None) else 0,
                "time_s":         elapsed,
            })
    return pd.DataFrame(rows)


def run_trace(scenario_fn, agent_a: str, agent_b: str, n_steps: int = 30) -> dict:
    """Run one negotiation per cross-combo for path visualisation."""
    s0, sf0, bf0 = scenario_fn(n_steps=n_steps)
    outcomes = _get_outcomes(s0)
    brv, srv = float(bf0.reserved_value or 0.0), float(sf0.reserved_value or 0.0)
    pareto0  = _pareto_frontier(outcomes, bf0, sf0) if outcomes else []
    nash0    = _nash_point(pareto0, brv, srv)

    traces = {}
    for buyer_key, seller_key in [(agent_a, agent_b), (agent_b, agent_a)]:
        session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
        session.add(_make_agent(buyer_key,  "buyer"),  ufun=buyer_ufun)
        session.add(_make_agent(seller_key, "seller"), ufun=seller_ufun)
        session.run()
        combo = f"{buyer_key} vs {seller_key}"
        traces[combo] = {
            "trace":     _extract_trace(session, buyer_ufun, seller_ufun),
            "pareto":    pareto0,
            "nash":      nash0,
            "buyer_rv":  brv,
            "seller_rv": srv,
        }
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def print_evaluation(df: pd.DataFrame, stage_name: str, scenario_name: str):
    agreed = df[df["agreed"]]

    print(f"\n[bold cyan]{'═'*60}[/bold cyan]")
    print(f"[bold cyan]  {stage_name}  ·  {scenario_name}[/bold cyan]")
    print(f"[bold cyan]{'═'*60}[/bold cyan]")

    print("\n[bold yellow]── Agreement Rates ──────────────────────────────────[/bold yellow]")
    agg = (
        df.groupby("combo")
        .agg(total=("agreed", "count"), agreements=("agreed", "sum"))
        .assign(rate=lambda x: x["agreements"] / x["total"] * 100)
    )
    agg["rate"] = agg["rate"].map("{:.1f}%".format)
    print(agg.to_string())

    print("\n[bold yellow]── Utility & Welfare (agreements only) ─────────────[/bold yellow]")
    uw = (
        agreed.groupby("combo")[["buyer_utility", "seller_utility",
                                  "social_welfare", "buyer_advantage"]]
        .mean().round(4)
        .sort_values("buyer_advantage", ascending=False)
    )
    print(uw.to_string())

    print("\n[bold yellow]── Distance Metrics (lower = better) ───────────────[/bold yellow]")
    dist = (
        agreed.groupby("combo")[["pareto_dist", "nash_dist"]]
        .mean().round(4)
        .sort_values("nash_dist")
    )
    print(dist.to_string())

    print("\n[bold yellow]── Rank Summary (Sec 2.4 criteria) ─────────────────[/bold yellow]")
    rank = pd.DataFrame({
        "agree_%":        df.groupby("combo")["agreed"].mean() * 100,
        "advantage":      agreed.groupby("combo")["buyer_advantage"].mean(),
        "social_welfare": agreed.groupby("combo")["social_welfare"].mean(),
        "nash_dist":      agreed.groupby("combo")["nash_dist"].mean(),
        "pareto_dist":    agreed.groupby("combo")["pareto_dist"].mean(),
    }).sort_values("advantage", ascending=False).round(4)
    rank["agree_%"] = rank["agree_%"].map("{:.1f}%".format)
    print(rank.to_string())
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Utility Space
# ─────────────────────────────────────────────────────────────────────────────

def plot_utility_space(scenario_fn, df: pd.DataFrame, title: str, n_steps: int = 30):
    session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
    outcomes = _get_outcomes(session)
    if not outcomes:
        return

    brv = float(buyer_ufun.reserved_value  or 0.0)
    srv = float(seller_ufun.reserved_value or 0.0)
    all_bu = [float(buyer_ufun(o))  for o in outcomes]
    all_su = [float(seller_ufun(o)) for o in outcomes]
    pareto = _pareto_frontier(outcomes, buyer_ufun, seller_ufun)
    nash   = _nash_point(pareto, brv, srv)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(all_bu, all_su, s=10, alpha=0.12, color="#bdbdbd", zorder=1,
               label="All outcomes")
    ax.fill_betweenx([srv, 1.05], brv, 1.05, alpha=0.06, color="green",
                     label="ZOPA region")

    if pareto:
        p_sorted = sorted(pareto, key=lambda x: x[1])
        ax.plot([p[1] for p in p_sorted], [p[2] for p in p_sorted],
                color="steelblue", linewidth=2.0, zorder=3, label="Pareto frontier")
        ax.scatter([p[1] for p in p_sorted], [p[2] for p in p_sorted],
                   s=30, color="steelblue", zorder=4)

    if nash:
        ax.scatter([nash[1]], [nash[2]], s=260, marker="D",
                   color="gold", edgecolors="black", linewidths=1.5,
                   zorder=6, label="Nash point")

    agreed = df[df["agreed"]]
    for combo in df["combo"].unique():
        grp = agreed[agreed["combo"] == combo]
        if grp.empty:
            continue
        buyer_key = combo.split(" vs ")[0]
        ax.scatter(grp["buyer_utility"], grp["seller_utility"],
                   s=55, alpha=0.65, color=COLORS.get(buyer_key, "#888"),
                   label=combo, zorder=5)

    ax.axvline(brv, color="#e53935", linestyle="--", linewidth=1.5,
               label=f"Buyer rv={brv:.2f}")
    ax.axhline(srv, color="#b71c1c", linestyle=":",  linewidth=1.5,
               label=f"Seller rv={srv:.2f}")

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Buyer Utility", fontsize=12)
    ax.set_ylabel("Seller Utility", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Concession Paths
# ─────────────────────────────────────────────────────────────────────────────

def plot_concession_paths(scenario_fn, stage_name: str, agent_a: str, agent_b: str,
                          scenario_name: str, n_steps: int = 30):
    traces = run_trace(scenario_fn, agent_a, agent_b, n_steps=n_steps)
    combos = list(traces.keys())

    fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 5),
                             sharex=True, sharey=True)
    if len(combos) == 1:
        axes = [axes]

    sample  = next(iter(traces.values()))
    pareto  = sample["pareto"]
    nash    = sample["nash"]
    brv     = sample["buyer_rv"]
    srv     = sample["seller_rv"]

    for ax, combo in zip(axes, combos):
        trace = traces[combo]["trace"]

        if pareto:
            p_sorted = sorted(pareto, key=lambda x: x[1])
            ax.plot([p[1] for p in p_sorted], [p[2] for p in p_sorted],
                    color="steelblue", linewidth=1.5, alpha=0.6, zorder=2,
                    label="Pareto frontier")

        if nash:
            ax.scatter([nash[1]], [nash[2]], s=180, marker="D",
                       color="gold", edgecolors="black", linewidths=1.2,
                       zorder=5, label="Nash")

        ax.axvline(brv, color="#e53935", linestyle="--", linewidth=1.2)
        ax.axhline(srv, color="#b71c1c", linestyle=":",  linewidth=1.2)

        if trace:
            bu_vals = [t[1] for t in trace]
            su_vals = [t[2] for t in trace]
            n_pts   = len(trace)
            cmap    = plt.cm.get_cmap("plasma")

            for i in range(n_pts - 1):
                frac = i / max(n_pts - 1, 1)
                ax.annotate("",
                    xy=(bu_vals[i + 1], su_vals[i + 1]),
                    xytext=(bu_vals[i], su_vals[i]),
                    arrowprops=dict(arrowstyle="->", color=cmap(frac), lw=1.3))

            ax.scatter(bu_vals, su_vals, c=range(n_pts), cmap="plasma",
                       s=35, zorder=4, vmin=0, vmax=max(n_pts - 1, 1))
            ax.scatter([bu_vals[0]],  [su_vals[0]],  s=120, marker="o",
                       color="lime",  edgecolors="black", zorder=6, label="Start")
            ax.scatter([bu_vals[-1]], [su_vals[-1]], s=120, marker="*",
                       color="white", edgecolors="black", zorder=6, label="End")

        ax.set_xlim(-0.02, 1.08)
        ax.set_ylim(-0.02, 1.08)
        ax.set_title(combo, fontweight="bold", fontsize=10)
        ax.set_xlabel("Buyer Utility")
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.set_ylabel("Seller Utility")
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(f"Concession Paths — {stage_name} · {scenario_name}\n"
                 "(light=early, dark=late)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Self vs Joint Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_self_vs_joint(all_stage_dfs: list, scenario_names: list[str]):
    """One row per stage, one column per scenario."""
    n_stages = len(all_stage_dfs)
    n_scen   = len(scenario_names)
    fig, axes = plt.subplots(n_stages, n_scen,
                             figsize=(6 * n_scen, 5 * n_stages),
                             squeeze=False)

    for row, (stage_label, stage_dfs) in enumerate(all_stage_dfs):
        for col, (sname, df) in enumerate(zip(scenario_names, stage_dfs)):
            ax     = axes[row][col]
            agreed = df[df["agreed"]]

            for combo in df["combo"].unique():
                grp   = agreed[agreed["combo"] == combo]
                if grp.empty:
                    continue
                buyer_key = combo.split(" vs ")[0]
                color = COLORS.get(buyer_key, "#888")
                ax.scatter(grp["buyer_advantage"], grp["social_welfare"],
                           s=28, alpha=0.4, color=color, zorder=3)
                mx, my = grp["buyer_advantage"].mean(), grp["social_welfare"].mean()
                ax.scatter([mx], [my], s=180, marker="D", color=color,
                           edgecolors="black", linewidths=1.2,
                           zorder=5, label=combo)
                ex = grp["buyer_advantage"].std() * 1.96
                ey = grp["social_welfare"].std()  * 1.96
                ax.add_patch(plt.matplotlib.patches.Ellipse(
                    (mx, my), width=ex * 2, height=ey * 2,
                    color=color, alpha=0.12, zorder=2))

            ax.set_xlabel("Buyer Advantage", fontsize=9)
            ax.set_ylabel("Social Welfare",  fontsize=9)
            title = f"{stage_label}\n{sname}" if col == 0 else sname
            ax.set_title(title, fontweight="bold", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Self-interest vs Joint Quality\n(diamonds=mean, ellipses=95% CI)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Radar Summary (all 3 agents head-to-head across both stages)
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar_summary(combined_df: pd.DataFrame, scenario_names: list[str]):
    """Show all 3 agents' aggregate performance on one radar per scenario."""
    METRICS = [
        ("agree_%",          "Agreement\nRate"),
        ("advantage",        "Buyer\nAdvantage"),
        ("social_welfare",   "Social\nWelfare"),
        ("nash_closeness",   "Nash\nCloseness"),
        ("pareto_closeness", "Pareto\nCloseness"),
    ]
    n_metrics = len(METRICS)
    angles  = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]

    n_scen = len(scenario_names)
    fig, axes = plt.subplots(1, n_scen, figsize=(6 * n_scen, 5),
                             subplot_kw=dict(polar=True))
    if n_scen == 1:
        axes = [axes]

    for ax, sname in zip(axes, scenario_names):
        df     = combined_df[combined_df["scenario"] == sname]
        agreed = df[df["agreed"]]

        # Per-agent aggregate (agent = buyer_key for self-play or cross)
        # We score each agent as a buyer across all combos where it played buyer
        agent_metrics = {}
        for agent in AGENTS:
            grp = agreed[agreed["buyer_key"] == agent]
            all_grp = df[df["buyer_key"] == agent]
            if all_grp.empty:
                continue
            agent_metrics[agent] = {
                "agree_%":          all_grp["agreed"].mean(),
                "advantage":        grp["buyer_advantage"].mean() if not grp.empty else 0,
                "social_welfare":   grp["social_welfare"].mean() / 2 if not grp.empty else 0,
                "nash_closeness":   1 - grp["nash_dist"].mean() if not grp.empty else 0,
                "pareto_closeness": 1 - grp["pareto_dist"].mean() if not grp.empty else 0,
            }

        all_keys = [m[0] for m in METRICS]
        mins = {k: min(v[k] for v in agent_metrics.values()) for k in all_keys}
        maxs = {k: max(v[k] for v in agent_metrics.values()) for k in all_keys}

        for agent, vals in agent_metrics.items():
            normed = [
                (vals[k] - mins[k]) / (maxs[k] - mins[k] + 1e-9) for k in all_keys
            ]
            normed += normed[:1]
            color = COLORS.get(agent, "#888")
            ax.plot(angles, normed, color=color, linewidth=2, label=agent)
            ax.fill(angles, normed, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[1] for m in METRICS], fontsize=9)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6, color="gray")
        ax.set_ylim(0, 1)
        ax.set_title(sname, fontweight="bold", fontsize=11, pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    fig.suptitle("Agent Evolution — Radar Summary\n"
                 "(axes normalised within scenario; outer edge = best)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Built-in NegMAS plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_builtin_negmas(scenario_fn, stage_name: str,
                        agent_a: str, agent_b: str,
                        scenario_name: str, n_steps: int = 30):
    """One built-in NegMAS plot per cross-combo (A-buyer/B-seller and vice versa)."""
    for buyer_key, seller_key in [(agent_a, agent_b), (agent_b, agent_a)]:
        session, seller_ufun, buyer_ufun = scenario_fn(n_steps=n_steps)
        # Instantiate directly with short names — bypasses _make_agent's key-based label
        buyer_agent  = AGENTS[buyer_key](name=f"{SHORT[buyer_key]}-B")
        seller_agent = AGENTS[seller_key](name=f"{SHORT[seller_key]}-S")
        session.add(buyer_agent,  ufun=buyer_ufun)
        session.add(seller_agent, ufun=seller_ufun)
        session.run()

        combo_label = (f"{SHORT[buyer_key]} (buyer)  vs  {SHORT[seller_key]} (seller)\n"
                       f"{stage_name} · {scenario_name}")
        try:
            fig = session.plot(show_reserved=True)
        except TypeError:
            fig = session.plot()

        if fig is not None:
            fig.suptitle(f"NegMAS Built-in\n{combo_label}",
                         fontweight="bold", fontsize=10)
        else:
            plt.suptitle(f"NegMAS Built-in\n{combo_label}",
                         fontweight="bold", fontsize=10)
        plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    N_RUNS  = 20
    N_STEPS = 30

    print("[bold green]╔══════════════════════════════════════════════════════╗[/bold green]")
    print("[bold green]║  Agent Evolution: Freq → Bayesian → Full             ║[/bold green]")
    print("[bold green]╚══════════════════════════════════════════════════════╝[/bold green]")

    scenario_names = list(SCENARIOS.keys())
    all_dfs_per_stage  = []   # [(stage_label, [df_scen1, df_scen2])]
    combined_rows      = []

    for stage_name, agent_a, agent_b in STAGES:
        print(f"\n[bold cyan]{'─'*55}[/bold cyan]")
        print(f"[bold cyan]  {stage_name}[/bold cyan]")
        stage_dfs = []
        for sname, fn in SCENARIOS.items():
            print(f"\n  [cyan]Running scenario: {sname}...[/cyan]")
            df = run_stage(stage_name, agent_a, agent_b, fn, sname,
                           n_runs=N_RUNS, n_steps=N_STEPS)
            stage_dfs.append(df)
            combined_rows.append(df)
            print_evaluation(df, stage_name, sname)
        all_dfs_per_stage.append((stage_name, stage_dfs))

    combined_df = pd.concat(combined_rows, ignore_index=True)

    # ── Cross-stage agent summary ────────────────────────────────────────────
    print("\n[bold cyan]═══ Overall Agent Summary (as buyer) ═══[/bold cyan]")
    agreed_all = combined_df[combined_df["agreed"]]
    for agent in AGENTS:
        grp  = agreed_all[agreed_all["buyer_key"] == agent]
        rate = 100 * combined_df[combined_df["buyer_key"] == agent]["agreed"].mean()
        print(
            f"[yellow]{agent:12}[/yellow]  "
            f"agree={rate:.1f}%  "
            f"adv={grp['buyer_advantage'].mean():.3f}  "
            f"welfare={grp['social_welfare'].mean():.3f}  "
            f"nash_dist={grp['nash_dist'].mean():.4f}  "
            f"pareto_dist={grp['pareto_dist'].mean():.4f}"
        )

    print("\n[bold green]Generating plots...[/bold green]")

    # Plot 1 – Utility Space (one per stage per scenario)
    for stage_name, agent_a, agent_b in STAGES:
        for sname, fn, df in zip(scenario_names, SCENARIOS.values(),
                                  next(s[1] for s in all_dfs_per_stage if s[0] == stage_name)):
            plot_utility_space(fn, df, f"{stage_name} · {sname}", n_steps=N_STEPS)

    # Plot 2 – Concession Paths (one per stage per scenario)
    for stage_name, agent_a, agent_b in STAGES:
        for sname, fn in zip(scenario_names, SCENARIOS.values()):
            print(f"  [dim]Trace: {stage_name} · {sname}[/dim]")
            plot_concession_paths(fn, stage_name, agent_a, agent_b, sname, N_STEPS)

    # Plot 3 – Self vs Joint (all stages and scenarios)
    plot_self_vs_joint(all_dfs_per_stage, scenario_names)

    # Plot 4 – Radar (all 3 agents together)
    plot_radar_summary(combined_df, scenario_names)

    # Plot 5 – Built-in NegMAS (cross-combos per stage per scenario)
    for stage_name, agent_a, agent_b in STAGES:
        for sname, fn in zip(scenario_names, SCENARIOS.values()):
            print(f"  [dim]NegMAS plot: {stage_name} · {sname}[/dim]")
            plot_builtin_negmas(fn, stage_name, agent_a, agent_b, sname, N_STEPS)

    plt.show()


if __name__ == "__main__":
    main()
