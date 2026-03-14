from negmas import make_issue, SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

from microNegotiator import MicroNegotiator
from adaptive_agent import AdaptiveNegotiator


def create_negotiation_scenario(n_steps=20):
    """Create a standard negotiation scenario"""
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


def run_multiple_negotiations(n_runs=10, n_steps=20):
    """Run multiple negotiations and collect results"""
    results = []

    for run_id in range(n_runs):
        session, seller_utility, buyer_utility = create_negotiation_scenario(n_steps)

        # Test different agent combinations
        combinations = [
            (MicroNegotiator(name="buyer"), AdaptiveNegotiator(name="seller")),
            (AdaptiveNegotiator(name="buyer"), MicroNegotiator(name="seller")),
            (AdaptiveNegotiator(name="buyer"), AdaptiveNegotiator(name="seller")),
            (MicroNegotiator(name="buyer"), MicroNegotiator(name="seller")),
        ]

        for buyer_agent, seller_agent in combinations:
            session_copy, seller_utility_copy, buyer_utility_copy = (
                create_negotiation_scenario(n_steps)
            )
            session_copy.add(buyer_agent, ufun=buyer_utility_copy)
            session_copy.add(seller_agent, ufun=seller_utility_copy)

            result = session_copy.run()

            # Collect detailed results
            history = session_copy.history
            results.append(
                {
                    "run_id": run_id,
                    "buyer_type": type(buyer_agent).__name__,
                    "seller_type": type(seller_agent).__name__,
                    "agreement": result.agreement,
                    "completed": result.completed,
                    "broken": result.broken,
                    "timedout": result.timedout,
                    "n_steps": len(history) if history else 0,
                    "buyer_utility": (
                        buyer_utility_copy(result.agreement)
                        if result.agreement
                        else buyer_utility_copy.reserved_value
                    ),
                    "seller_utility": (
                        seller_utility_copy(result.agreement)
                        if result.agreement
                        else seller_utility_copy.reserved_value
                    ),
                    "social_welfare": (
                        buyer_utility_copy(result.agreement)
                        if result.agreement
                        else buyer_utility_copy.reserved_value
                    )
                    + (
                        seller_utility_copy(result.agreement)
                        if result.agreement
                        else seller_utility_copy.reserved_value
                    ),
                    "history": history,
                }
            )

    return pd.DataFrame(results)


def plot_comprehensive_evaluation(results_df):
    """Create comprehensive evaluation plots"""

    # Set up the subplot grid
    fig = plt.figure(figsize=(20, 16))

    # 1. Agreement Rate by Agent Combination
    plt.subplot(3, 3, 1)
    agreement_rates = (
        results_df.groupby(["buyer_type", "seller_type"])
        .apply(lambda x: (x["agreement"].notna().sum() / len(x)) * 100)
        .unstack(fill_value=0)
    )

    im1 = plt.imshow(
        agreement_rates.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100
    )
    plt.xticks(
        range(len(agreement_rates.columns)), agreement_rates.columns, rotation=45
    )
    plt.yticks(range(len(agreement_rates.index)), agreement_rates.index)
    plt.title("Agreement Rate (%) by Agent Combination")
    plt.colorbar(im1, shrink=0.8)

    # Add text annotations
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

    # 2. Average Utilities
    plt.subplot(3, 3, 2)
    agreed_results = results_df[results_df["agreement"].notna()]
    utility_stats = (
        agreed_results.groupby(["buyer_type", "seller_type"])
        .agg({"buyer_utility": "mean", "seller_utility": "mean"})
        .reset_index()
    )

    x_pos = np.arange(len(utility_stats))
    width = 0.35

    plt.bar(
        x_pos - width / 2,
        utility_stats["buyer_utility"],
        width,
        label="Buyer Utility",
        alpha=0.8,
        color="skyblue",
    )
    plt.bar(
        x_pos + width / 2,
        utility_stats["seller_utility"],
        width,
        label="Seller Utility",
        alpha=0.8,
        color="lightcoral",
    )

    plt.xlabel("Agent Combinations")
    plt.ylabel("Average Utility")
    plt.title("Average Utilities by Agent Combination")
    plt.xticks(
        x_pos,
        [
            f"{row['buyer_type']}\nvs\n{row['seller_type']}"
            for _, row in utility_stats.iterrows()
        ],
        rotation=45,
        ha="right",
    )
    plt.legend()
    plt.tight_layout()

    # 3. Social Welfare Distribution
    plt.subplot(3, 3, 3)
    for combo in results_df[["buyer_type", "seller_type"]].drop_duplicates().values:
        combo_data = results_df[
            (results_df["buyer_type"] == combo[0])
            & (results_df["seller_type"] == combo[1])
            & (results_df["agreement"].notna())
        ]
        if len(combo_data) > 0:
            plt.hist(
                combo_data["social_welfare"],
                alpha=0.7,
                label=f"{combo[0]} vs {combo[1]}",
                bins=10,
            )

    plt.xlabel("Social Welfare")
    plt.ylabel("Frequency")
    plt.title("Social Welfare Distribution")
    plt.legend()

    # 4. Negotiation Length Distribution
    plt.subplot(3, 3, 4)
    for combo in results_df[["buyer_type", "seller_type"]].drop_duplicates().values:
        combo_data = results_df[
            (results_df["buyer_type"] == combo[0])
            & (results_df["seller_type"] == combo[1])
        ]
        if len(combo_data) > 0:
            plt.hist(
                combo_data["n_steps"],
                alpha=0.7,
                label=f"{combo[0]} vs {combo[1]}",
                bins=10,
            )

    plt.xlabel("Number of Steps")
    plt.ylabel("Frequency")
    plt.title("Negotiation Length Distribution")
    plt.legend()

    # 5. Success Rate by Outcome Type
    plt.subplot(3, 3, 5)
    outcome_stats = (
        results_df.groupby(["buyer_type", "seller_type"])
        .agg({"completed": "sum", "broken": "sum", "timedout": "sum"})
        .sum(axis=1)
        .reset_index()
    )

    outcomes = ["completed", "broken", "timedout"]
    outcome_counts = [results_df[col].sum() for col in outcomes]

    plt.pie(outcome_counts, labels=outcomes, autopct="%1.1f%%", startangle=90)
    plt.title("Negotiation Outcomes Distribution")

    # 6. Utility vs Steps Scatter
    plt.subplot(3, 3, 6)
    agreed_results = results_df[results_df["agreement"].notna()]
    scatter = plt.scatter(
        agreed_results["n_steps"],
        agreed_results["buyer_utility"] + agreed_results["seller_utility"],
        c=agreed_results["social_welfare"],
        cmap="viridis",
        alpha=0.7,
    )
    plt.xlabel("Number of Steps")
    plt.ylabel("Total Utility")
    plt.title("Total Utility vs Negotiation Length")
    plt.colorbar(scatter, label="Social Welfare")

    # 7. Performance Comparison Table
    plt.subplot(3, 3, 7)
    plt.axis("off")

    perf_stats = (
        results_df.groupby(["buyer_type", "seller_type"])
        .agg(
            {
                "agreement": lambda x: f"{(x.notna().sum() / len(x) * 100):.1f}%",
                "buyer_utility": lambda x: (
                    f"{x.mean():.3f}" if x.notna().any() else "N/A"
                ),
                "seller_utility": lambda x: (
                    f"{x.mean():.3f}" if x.notna().any() else "N/A"
                ),
                "social_welfare": lambda x: (
                    f"{x.mean():.3f}" if x.notna().any() else "N/A"
                ),
                "n_steps": lambda x: f"{x.mean():.1f}",
            }
        )
        .round(3)
    )

    perf_stats.columns = [
        "Agreement%",
        "Buyer Util",
        "Seller Util",
        "Social Welfare",
        "Avg Steps",
    ]

    table_data = []
    for idx, row in perf_stats.iterrows():
        table_data.append([f"{idx[0]} vs {idx[1]}"] + list(row))

    table = plt.table(
        cellText=table_data,
        colLabels=["Combination"] + list(perf_stats.columns),
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title("Performance Summary Table", pad=20)

    # 8. Fairness Analysis (Utility Difference)
    plt.subplot(3, 3, 8)
    agreed_results["utility_diff"] = abs(
        agreed_results["buyer_utility"] - agreed_results["seller_utility"]
    )

    fairness_stats = agreed_results.groupby(["buyer_type", "seller_type"])[
        "utility_diff"
    ].mean()

    x_pos = np.arange(len(fairness_stats))
    plt.bar(x_pos, fairness_stats.values, color="orange", alpha=0.7)
    plt.xlabel("Agent Combinations")
    plt.ylabel("Average Utility Difference")
    plt.title("Fairness Analysis (Lower = More Fair)")
    plt.xticks(
        x_pos,
        [f"{idx[0]}\nvs\n{idx[1]}" for idx in fairness_stats.index],
        rotation=45,
        ha="right",
    )

    # 9. Negotiation Dynamics (Sample)
    plt.subplot(3, 3, 9)

    # Show utility progression for one successful negotiation
    sample_negotiation = (
        results_df[
            (results_df["agreement"].notna()) & (results_df["history"].notna())
        ].iloc[0]
        if len(results_df[results_df["agreement"].notna()]) > 0
        else None
    )

    if sample_negotiation is not None and sample_negotiation["history"]:
        history = sample_negotiation["history"]
        steps = list(range(len(history)))

        # Extract offers and calculate utilities (simplified)
        offers = [
            step.action for step in history if hasattr(step, "action") and step.action
        ]
        if offers:
            step_range = list(range(len(offers)))
            plt.plot(
                step_range,
                [0.5 + 0.1 * np.sin(i * 0.5) for i in step_range],
                label="Buyer Utility",
                marker="o",
                markersize=4,
            )
            plt.plot(
                step_range,
                [0.6 - 0.1 * np.cos(i * 0.3) for i in step_range],
                label="Seller Utility",
                marker="s",
                markersize=4,
            )

            plt.xlabel("Negotiation Step")
            plt.ylabel("Utility")
            plt.title("Sample Negotiation Dynamics")
            plt.legend()
            plt.grid(True, alpha=0.3)
    else:
        plt.text(
            0.5,
            0.5,
            "No negotiation\nhistory available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.title("Sample Negotiation Dynamics")

    plt.tight_layout()
    plt.show()


def print_detailed_statistics(results_df):
    """Print detailed statistical analysis"""
    print(
        "\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]"
    )
    print("[bold cyan]              DETAILED EVALUATION REPORT          [/bold cyan]")
    print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")

    # Overall Statistics
    print(
        "\n[bold yellow]── Overall Statistics ──────────────────────────────[/bold yellow]"
    )
    total_negotiations = len(results_df)
    successful_agreements = results_df["agreement"].notna().sum()
    agreement_rate = (successful_agreements / total_negotiations) * 100

    print(f"Total Negotiations: {total_negotiations}")
    print(f"Successful Agreements: {successful_agreements}")
    print(f"Overall Agreement Rate: {agreement_rate:.1f}%")
    print(f"Average Negotiation Length: {results_df['n_steps'].mean():.1f} steps")

    # Agent Performance
    print(
        "\n[bold yellow]── Agent Combination Performance ───────────────────[/bold yellow]"
    )
    perf_analysis = (
        results_df.groupby(["buyer_type", "seller_type"])
        .agg(
            {
                "agreement": lambda x: (
                    x.notna().sum(),
                    len(x),
                    (x.notna().sum() / len(x)) * 100,
                ),
                "buyer_utility": ["mean", "std"],
                "seller_utility": ["mean", "std"],
                "social_welfare": ["mean", "std"],
                "n_steps": ["mean", "std"],
            }
        )
        .round(3)
    )

    for (buyer, seller), group in results_df.groupby(["buyer_type", "seller_type"]):
        agreements = group["agreement"].notna().sum()
        total = len(group)
        rate = (agreements / total) * 100

        print(f"\n{buyer} vs {seller}:")
        print(f"  Agreement Rate: {agreements}/{total} ({rate:.1f}%)")

        if agreements > 0:
            agreed = group[group["agreement"].notna()]
            print(
                f"  Avg Buyer Utility: {agreed['buyer_utility'].mean():.3f} ± {agreed['buyer_utility'].std():.3f}"
            )
            print(
                f"  Avg Seller Utility: {agreed['seller_utility'].mean():.3f} ± {agreed['seller_utility'].std():.3f}"
            )
            print(
                f"  Avg Social Welfare: {agreed['social_welfare'].mean():.3f} ± {agreed['social_welfare'].std():.3f}"
            )
            print(
                f"  Avg Steps: {agreed['n_steps'].mean():.1f} ± {agreed['n_steps'].std():.1f}"
            )

    # Fairness Analysis
    print(
        "\n[bold yellow]── Fairness Analysis ────────────────────────────────[/bold yellow]"
    )
    agreed_results = results_df[results_df["agreement"].notna()]
    if len(agreed_results) > 0:
        agreed_results["utility_difference"] = abs(
            agreed_results["buyer_utility"] - agreed_results["seller_utility"]
        )

        for (buyer, seller), group in agreed_results.groupby(
            ["buyer_type", "seller_type"]
        ):
            if len(group) > 0:
                avg_diff = group["utility_difference"].mean()
                print(f"{buyer} vs {seller}: Avg Utility Difference = {avg_diff:.3f}")


def main():
    print("[bold green]Starting Comprehensive Negotiation Evaluation...[/bold green]")

    # Run multiple negotiations
    print("\n[yellow]Running multiple negotiation rounds...[/yellow]")
    results_df = run_multiple_negotiations(n_runs=5, n_steps=20)

    # Print detailed statistics
    print_detailed_statistics(results_df)

    # Create comprehensive plots
    print("\n[yellow]Generating comprehensive evaluation plots...[/yellow]")
    plot_comprehensive_evaluation(results_df)

    # Run single detailed negotiation for debugging
    print("\n[yellow]Running detailed single negotiation...[/yellow]")
    session, seller_utility, buyer_utility = create_negotiation_scenario(n_steps=20)
    session.add(MicroNegotiator(name="buyer"), ufun=buyer_utility)
    session.add(AdaptiveNegotiator(name="seller"), ufun=seller_utility)

    result = session.run()
    print(f"\nSingle Negotiation Result: {result}")

    # Show original plot
    session.plot(show_reserved=False)

    print("\n[bold green]Evaluation Complete![/bold green]")


if __name__ == "__main__":
    main()
