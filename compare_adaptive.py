from pathlib import Path
import multiprocessing as mp
import time

import numpy as np
import plotly.graph_objects as go
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


def main():
    tic = time.perf_counter()
    path = Path.home() / "negmas" / unique_name("group31_test")

    results = cartesian_tournament(
        competitors=[
            AdaptiveNegotiator,
            BoulwareTBNegotiator,
            ConcederTBNegotiator,
            LinearTBNegotiator,
        ],
        scenarios=get_scenarios(n=2),
        n_repetitions=10,
        path=path,
        njobs=-1,  # Serialaize
    )

    plot_kde(results)


if __name__ == "__main__":
    mp.freeze_support()
    main()
