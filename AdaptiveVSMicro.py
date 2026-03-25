from negmas import make_issue, SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from microNegotiator import MicroNegotiator
from adaptive_boulware_mitigate import AdaptiveBoulwareMitigate


# -----------------------------------------------------------------------------
# Scenario setup
# -----------------------------------------------------------------------------


def create_negotiation_scenario(n_steps=20):
    """Create a standard 3-issue negotiation scenario."""
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
        reserved_value=0.4,
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

    return session, seller_utility, buyer_utility


def create_single_issue_scenario(n_steps=20):
    issues = [make_issue(name="price", values=13)]  # 0..12
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    seller_min_price = 6
    buyer_max_price = 8

    seller_reserved_utility = seller_min_price / 12.0
    buyer_reserved_utility = (12.0 - buyer_max_price) / 12.0

    seller_utility = LUFun(
        values={"price": IdentityFun()},
        outcome_space=session.outcome_space,
        reserved_value=seller_reserved_utility,
    ).scale_max(1.0)

    buyer_utility = LUFun(
        values={"price": AffineFun(-1, bias=12.0)},
        outcome_space=session.outcome_space,
        reserved_value=buyer_reserved_utility,
    ).scale_max(1.0)

    return session, seller_utility, buyer_utility


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _combo_label(row):
    return f"{row['buyer_type']} vs {row['seller_type']}"


def _agreed_results(results_df):
    return results_df[results_df["agreement"].notna()].copy()


def _safe_offer_tuple(offer):
    if offer is None:
        return None
    try:
        return tuple(offer)
    except TypeError:
        return offer


def _offer_to_text(offer):
    offer = _safe_offer_tuple(offer)
    if offer is None:
        return "None"
    try:
        if len(offer) == 1:
            return f"price={offer[0]}"
        if len(offer) >= 3:
            return f"p={offer[0]}, q={offer[1]}, d={offer[2]}"
    except Exception:
        pass
    return str(offer)


def _get_all_outcomes(session):
    """Best-effort extraction of all discrete outcomes from NegMAS."""
    if hasattr(session, "outcomes") and session.outcomes is not None:
        try:
            return [_safe_offer_tuple(o) for o in session.outcomes]
        except Exception:
            pass

    os = getattr(session, "outcome_space", None)
    if os is not None:
        for name in ("enumerate_or_sample", "enumerate", "all"):
            if hasattr(os, name):
                attr = getattr(os, name)
                try:
                    values = attr() if callable(attr) else attr
                    return [_safe_offer_tuple(o) for o in values]
                except Exception:
                    continue

    raise RuntimeError("Could not enumerate outcomes from the session")


def _pareto_mask(points):
    """Return boolean mask of Pareto-efficient rows for max-max utilities."""
    vals = points[["buyer_utility", "seller_utility"]].to_numpy(dtype=float)
    n = len(vals)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        x = vals[i]
        dominates_i = (
            (vals[:, 0] >= x[0])
            & (vals[:, 1] >= x[1])
            & ((vals[:, 0] > x[0]) | (vals[:, 1] > x[1]))
        )
        if dominates_i.any():
            mask[i] = False
    return mask


def _nash_index(points, buyer_rv, seller_rv):
    gains = (points["buyer_utility"] - buyer_rv).clip(lower=0) * (
        points["seller_utility"] - seller_rv
    ).clip(lower=0)
    if len(gains) == 0:
        return None
    return int(gains.idxmax())


def _extract_offer_from_history_entry(entry):
    """Best-effort extraction of an offer from a NegMAS history entry."""
    if entry is None:
        return None

    for name in (
        "current_offer",
        "offer",
        "agreement",
        "proposal",
        "outcome",
        "new_offer",
    ):
        if hasattr(entry, name):
            offer = getattr(entry, name)
            if offer is not None:
                return _safe_offer_tuple(offer)

    action = getattr(entry, "action", None)
    if action is not None:
        for name in (
            "current_offer",
            "offer",
            "agreement",
            "proposal",
            "outcome",
            "new_offer",
        ):
            if hasattr(action, name):
                offer = getattr(action, name)
                if offer is not None:
                    return _safe_offer_tuple(offer)
        try:
            return _safe_offer_tuple(action)
        except Exception:
            pass

    if isinstance(entry, dict):
        for name in (
            "current_offer",
            "offer",
            "agreement",
            "proposal",
            "outcome",
            "new_offer",
            "action",
        ):
            if name in entry and entry[name] is not None:
                return _safe_offer_tuple(entry[name])

    return None


def build_single_session_summary(session, buyer_utility, seller_utility, result):
    """Create a DataFrame with utility-space coordinates for all outcomes."""
    outcomes = _get_all_outcomes(session)
    df = pd.DataFrame(
        {
            "offer": outcomes,
            "buyer_utility": [float(buyer_utility(o)) for o in outcomes],
            "seller_utility": [float(seller_utility(o)) for o in outcomes],
        }
    )
    df["social_welfare"] = df["buyer_utility"] + df["seller_utility"]
    df["label"] = df["offer"].apply(_offer_to_text)
    df["is_pareto"] = _pareto_mask(df)

    buyer_rv = float(buyer_utility.reserved_value or 0.0)
    seller_rv = float(seller_utility.reserved_value or 0.0)
    nash_idx = _nash_index(df, buyer_rv, seller_rv)
    df["is_nash"] = False
    if nash_idx is not None:
        df.loc[nash_idx, "is_nash"] = True

    agreement = _safe_offer_tuple(getattr(result, "agreement", None))
    df["is_agreement"] = df["offer"] == agreement if agreement is not None else False
    return df


# -----------------------------------------------------------------------------
# Evaluation runs
# -----------------------------------------------------------------------------


def run_multiple_negotiations(n_runs=10, n_steps=20):
    """Run multiple negotiations and collect results."""
    results = []

    for run_id in range(n_runs):
        combinations = [
            (MicroNegotiator(name="buyer"), AdaptiveBoulwareMitigate(name="seller")),
            (AdaptiveBoulwareMitigate(name="buyer"), MicroNegotiator(name="seller")),
            (AdaptiveBoulwareMitigate(name="buyer"), AdaptiveBoulwareMitigate(name="seller")),
            (MicroNegotiator(name="buyer"), MicroNegotiator(name="seller")),
        ]

        for buyer_agent, seller_agent in combinations:
            session, seller_utility, buyer_utility = create_single_issue_scenario(
                n_steps
            )
            session.add(buyer_agent, ufun=buyer_utility)
            session.add(seller_agent, ufun=seller_utility)
            result = session.run()

            agreement = _safe_offer_tuple(getattr(result, "agreement", None))
            agreed = agreement is not None

            buyer_u = (
                float(buyer_utility(agreement))
                if agreed
                else float(buyer_utility.reserved_value)
            )
            seller_u = (
                float(seller_utility(agreement))
                if agreed
                else float(seller_utility.reserved_value)
            )

            results.append(
                {
                    "run_id": run_id,
                    "buyer_type": type(buyer_agent).__name__,
                    "seller_type": type(seller_agent).__name__,
                    "combo": f"{type(buyer_agent).__name__} vs {type(seller_agent).__name__}",
                    "agreement": agreement,
                    "agreed": agreed,
                    "completed": bool(getattr(result, "completed", False)),
                    "broken": bool(getattr(result, "broken", False)),
                    "timedout": bool(getattr(result, "timedout", False)),
                    "n_steps": (
                        len(session.history) if getattr(session, "history", None) else 0
                    ),
                    "buyer_utility": buyer_u,
                    "seller_utility": seller_u,
                    "social_welfare": buyer_u + seller_u,
                    "utility_diff": abs(buyer_u - seller_u),
                    "history": getattr(session, "history", None),
                }
            )

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Aggregate plots
# -----------------------------------------------------------------------------


def plot_agreement_rate(results_df):
    plt.figure("Agreement Rate", figsize=(8, 6))
    agreement_rates = (
        results_df.groupby(["buyer_type", "seller_type"])["agreed"]
        .mean()
        .mul(100)
        .unstack(fill_value=0)
    )

    im = plt.imshow(
        agreement_rates.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100
    )
    plt.xticks(
        range(len(agreement_rates.columns)),
        agreement_rates.columns,
        rotation=30,
        ha="right",
    )
    plt.yticks(range(len(agreement_rates.index)), agreement_rates.index)
    plt.title("Agreement Rate (%)")
    plt.colorbar(im, shrink=0.8)

    for i in range(len(agreement_rates.index)):
        for j in range(len(agreement_rates.columns)):
            plt.text(
                j,
                i,
                f"{agreement_rates.iloc[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )
    plt.tight_layout()


def plot_outcomes_by_combination(results_df):
    plt.figure("Outcomes", figsize=(10, 6))
    summary = results_df.groupby("combo")[["completed", "broken", "timedout"]].sum()
    x = np.arange(len(summary))

    completed = summary["completed"].values
    broken = summary["broken"].values
    timedout = summary["timedout"].values

    plt.bar(x, completed, label="completed")
    plt.bar(x, broken, bottom=completed, label="broken")
    plt.bar(x, timedout, bottom=completed + broken, label="timedout")
    plt.xticks(x, summary.index, rotation=25, ha="right")
    plt.ylabel("Count")
    plt.title("Outcome Counts by Combination")
    plt.legend()
    plt.tight_layout()


def plot_performance_table(results_df):
    plt.figure("Performance Table", figsize=(12, 4.8))
    plt.axis("off")

    agreed = _agreed_results(results_df)
    agreement_rate = results_df.groupby("combo")["agreed"].mean().mul(100)
    avg_steps = results_df.groupby("combo")["n_steps"].mean()

    table_df = pd.DataFrame(
        {
            "Agreement %": agreement_rate.map(lambda x: f"{x:.1f}%"),
            "Avg Buyer Util": agreed.groupby("combo")["buyer_utility"].mean().round(3),
            "Avg Seller Util": agreed.groupby("combo")["seller_utility"]
            .mean()
            .round(3),
            "Avg Welfare": agreed.groupby("combo")["social_welfare"].mean().round(3),
            "Avg Steps": avg_steps.round(1),
        }
    ).fillna("-")

    table = plt.table(
        cellText=table_df.reset_index().values,
        colLabels=["Combination"] + list(table_df.columns),
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    plt.title("Performance Summary")
    plt.tight_layout()


def plot_sample_negotiation_dynamics(results_df, buyer_utility, seller_utility):
    """Plot real offer utilities from one sample history."""
    plt.figure("Sample Dynamics", figsize=(10, 6))
    sample = results_df[results_df["history"].notna()]
    if sample.empty:
        plt.text(0.5, 0.5, "No negotiation history available", ha="center", va="center")
        plt.title("Sample Negotiation Dynamics")
        plt.tight_layout()
        return

    history = sample.iloc[0]["history"]
    offers = []
    for entry in history:
        offer = _extract_offer_from_history_entry(entry)
        if offer is not None:
            offers.append(offer)

    if not offers:
        plt.text(
            0.5, 0.5, "Could not extract offers from history", ha="center", va="center"
        )
        plt.title("Sample Negotiation Dynamics")
        plt.tight_layout()
        return

    buyer_trace = [float(buyer_utility(o)) for o in offers]
    seller_trace = [float(seller_utility(o)) for o in offers]
    x = np.arange(len(offers))

    plt.plot(x, buyer_trace, marker="o", label="Buyer utility of each offer")
    plt.plot(x, seller_trace, marker="s", label="Seller utility of each offer")
    plt.xlabel("Offer index")
    plt.ylabel("Utility")
    plt.title("Sample Negotiation Dynamics (Real Utilities)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


# -----------------------------------------------------------------------------
# Single-session custom utility map
# -----------------------------------------------------------------------------


def plot_single_session_utility_map(
    session,
    buyer_utility,
    seller_utility,
    result,
    scale_to_100=False,
    annotate_frontier_points=8,
    title="Negotiation Utility Map",
):
    """
    Plot:
    - all outcomes in utility space
    - Pareto frontier
    - reservation lines
    - Nash point
    - agreement point
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    df = build_single_session_summary(session, buyer_utility, seller_utility, result)

    scale = 100.0 if scale_to_100 else 1.0
    bx = df["buyer_utility"] * scale
    sy = df["seller_utility"] * scale

    ax.scatter(bx, sy, s=28, alpha=0.30, label="All outcomes")

    frontier = df[df["is_pareto"]].copy().sort_values("buyer_utility")
    ax.scatter(
        frontier["buyer_utility"] * scale,
        frontier["seller_utility"] * scale,
        s=70,
        label="Pareto frontier",
    )
    ax.plot(
        frontier["buyer_utility"] * scale,
        frontier["seller_utility"] * scale,
        linewidth=1.5,
        alpha=0.9,
    )

    if annotate_frontier_points > 0 and not frontier.empty:
        idxs = np.linspace(
            0,
            len(frontier) - 1,
            min(annotate_frontier_points, len(frontier)),
            dtype=int,
        )
        frontier_subset = frontier.iloc[np.unique(idxs)]
        for _, row in frontier_subset.iterrows():
            ax.annotate(
                row["label"],
                (row["buyer_utility"] * scale, row["seller_utility"] * scale),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )

    buyer_rv = float(buyer_utility.reserved_value or 0.0) * scale
    seller_rv = float(seller_utility.reserved_value or 0.0) * scale
    ax.axvline(buyer_rv, color="red", linewidth=2, label="Buyer reservation")
    ax.axhline(seller_rv, color="darkred", linewidth=2, label="Seller reservation")

    nash = df[df["is_nash"]]
    if not nash.empty:
        row = nash.iloc[0]
        ax.scatter(
            [row["buyer_utility"] * scale],
            [row["seller_utility"] * scale],
            s=180,
            marker="D",
            label="Nash point",
        )
        ax.annotate(
            "Nash",
            (row["buyer_utility"] * scale, row["seller_utility"] * scale),
            xytext=(8, -14),
            textcoords="offset points",
        )

    agreement = df[df["is_agreement"]]
    if not agreement.empty:
        row = agreement.iloc[0]
        ax.scatter(
            [row["buyer_utility"] * scale],
            [row["seller_utility"] * scale],
            s=260,
            marker="*",
            label="Agreement",
        )
        ax.annotate(
            f"Agreement\n{row['label']}",
            (row["buyer_utility"] * scale, row["seller_utility"] * scale),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
        )

    xmax = max(1.0 * scale, bx.max() * 1.08 if len(bx) else 1.0)
    ymax = max(1.0 * scale, sy.max() * 1.08 if len(sy) else 1.0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Utility of buyer" + (" (0-100)" if scale_to_100 else ""))
    ax.set_ylabel("Utility of seller" + (" (0-100)" if scale_to_100 else ""))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()


# -----------------------------------------------------------------------------
# Driver helpers
# -----------------------------------------------------------------------------


def plot_custom_aggregate_evaluation(results_df, buyer_utility, seller_utility):
    plot_agreement_rate(results_df)
    plot_outcomes_by_combination(results_df)
    plot_performance_table(results_df)
    plot_sample_negotiation_dynamics(results_df, buyer_utility, seller_utility)
    plt.show()


def print_detailed_statistics(results_df):
    print(
        "\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]"
    )
    print("[bold cyan]              DETAILED EVALUATION REPORT          [/bold cyan]")
    print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")

    total = len(results_df)
    agreements = int(results_df["agreed"].sum())
    print(f"\nTotal Negotiations: {total}")
    print(f"Successful Agreements: {agreements}")
    print(f"Overall Agreement Rate: {100.0 * agreements / max(total, 1):.1f}%")
    print(f"Average Negotiation Length: {results_df['n_steps'].mean():.1f} steps")

    print(
        "\n[bold yellow]── Agent Combination Performance ───────────────────[/bold yellow]"
    )
    for combo, group in results_df.groupby("combo"):
        agreed = group[group["agreed"]].copy()
        rate = 100.0 * group["agreed"].mean()
        print(f"\n{combo}:")
        print(f"  Agreement Rate: {rate:.1f}%")
        print(
            f"  Avg Steps: {group['n_steps'].mean():.1f} ± {group['n_steps'].std():.1f}"
        )
        if not agreed.empty:
            print(
                f"  Avg Buyer Utility: {agreed['buyer_utility'].mean():.3f} ± {agreed['buyer_utility'].std():.3f}"
            )
            print(
                f"  Avg Seller Utility: {agreed['seller_utility'].mean():.3f} ± {agreed['seller_utility'].std():.3f}"
            )
            print(
                f"  Avg Social Welfare: {agreed['social_welfare'].mean():.3f} ± {agreed['social_welfare'].std():.3f}"
            )
            print(f"  Avg Utility Difference: {agreed['utility_diff'].mean():.3f}")


def plot_builtin_negmas_session(session):
    """Use NegMAS built-in plotting for a single session."""
    try:
        session.plot(show_reserved=True)
    except TypeError:
        session.plot()


def print_builtin_negmas_analysis(session):
    """Print built-in NegMAS analytics if available."""
    try:
        frontier = session.pareto_frontier()
        print(f"[green]Built-in Pareto frontier:[/green] {frontier}")
    except Exception as e:
        print(f"[red]Could not get built-in Pareto frontier:[/red] {e}")

    try:
        nash = session.nash_points()
        print(f"[green]Built-in Nash points:[/green] {nash}")
    except Exception as e:
        print(f"[red]Could not get built-in Nash points:[/red] {e}")


def main():
    print("[bold green]Starting improved negotiation evaluation...[/bold green]")

    # Aggregate evaluation
    _, seller_utility_ref, buyer_utility_ref = create_single_issue_scenario(n_steps=20)
    results_df = run_multiple_negotiations(n_runs=5, n_steps=20)
    print_detailed_statistics(results_df)

    print("\n[yellow]Generating custom aggregate evaluation plots...[/yellow]")
    plot_custom_aggregate_evaluation(results_df, buyer_utility_ref, seller_utility_ref)

    # Single session
    print("\n[yellow]Running single detailed negotiation...[/yellow]")
    session, seller_utility, buyer_utility = create_single_issue_scenario(n_steps=20)
    session.add(MicroNegotiator(name="buyer"), ufun=buyer_utility)
    session.add(MicroNegotiator(name="seller"), ufun=seller_utility)

    result = session.run()
    print(f"Result: {result}")

    # Built-in NegMAS plot
    print("\n[yellow]Showing built-in NegMAS session plot...[/yellow]")
    plot_builtin_negmas_session(session)

    # Built-in analytics
    print("\n[yellow]Printing built-in NegMAS analysis...[/yellow]")
    print_builtin_negmas_analysis(session)

    # Custom combined utility map
    print("\n[yellow]Showing custom utility map...[/yellow]")
    plot_single_session_utility_map(
        session,
        buyer_utility,
        seller_utility,
        result,
        scale_to_100=False,
        title="Utility Map: MicroNegotiator vs AdaptiveBoulwareMitigate",
    )
    plt.show()


if __name__ == "__main__":
    main()
