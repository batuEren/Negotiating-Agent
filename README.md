# Group 31 Negotiating Agent

## Main Agent

**`group31_agent.py`** — `Group31_Negotiator`

Our submission agent. Implements a full adaptive SAO negotiator with:

- Bayesian opponent model (Bayes' rule)
- PrONeg outcome prediction (time-series forecasting + Monte Carlo)
- Adaptive target utility with Boulware backstop
- Boulware opponent detection and mitigation

---

## Folder Structure

```
Negotiating-Agent/
│
├── group31_agent.py                  # MAIN AGENT — Group31_Negotiator
│
├── Development versions (older)
│   ├── adaptive_frequency.py         # v1 — Frequency Analysis opponent model
│   ├── adaptive_bayesian.py          # v2 — Bayesian opponent model
│   └── adaptive_proneg.py            # v3 — Bayesian + PrONeg (without Boulware mitigation)
│
├── Baseline agents
│   ├── microNegotiator.py            # MiCRO strategy (no opponent modeling)
│   ├── timeBasedAgent.py             # Simple time-based conceding agent
│   └── titTat.py                     # Tit-for-Tat agent
│
├── Evaluation & comparison scripts
│   ├── AgentEvolution.py             # Compares v1 → v2 → v3 to track improvement over versions
│   ├── AdaptiveVSBoulware.py         # Head-to-head: Group31_Negotiator vs BoulwareTBNegotiator
│   ├── AdaptiveVSMicro.py            # Head-to-head: Group31_Negotiator vs MicroNegotiator
│   ├── tournament_group31_agent.py      # Benchmarks Group31_Negotiator against NegMAS built-ins
│   ├── tournament_adaptive_bayesian.py  # Benchmarks AdaptiveBayesian against NegMAS built-ins
│   ├── tournament_adaptive_frequency.py # Benchmarks AdaptiveFrequency against NegMAS built-ins
│   ├── tournament_adaptive_proneg.py    # Benchmarks AdaptivePrONeg against NegMAS built-ins
│   ├── comprehensive_evaluation.py   # Full multi-scenario evaluation across all agent combinations
│   ├── evaluation_comparison.py      # Head-to-head utility/welfare/Pareto metrics
│   └── evaluation_2.py               # Additional evaluation script
│
└── Other
    ├── negmasTest.py                 # Minimal NegMAS smoke test / sandbox
    ├── requirements.txt              # Python dependencies
    └── comparison_results.png        # Saved plot output
```

---

## Agent Versions

| Agent                    | File                    | Opponent Model     | Extra Strategy               |
|--------------------------|-------------------------|--------------------|------------------------------|
| `AdaptiveFrequency`      | `adaptive_frequency.py` | Frequency Analysis | —                            |
| `AdaptiveBayesian`       | `adaptive_bayesian.py`  | Bayesian           | —                            |
| `AdaptivePrONeg`         | `adaptive_proneg.py`    | Bayesian           | PrONeg outcome prediction    |
| **`Group31_Negotiator`** | **`group31_agent.py`**  | Bayesian           | PrONeg + Boulware mitigation |

All adaptive agents share:

- Adaptive target utility with backstop curve
- AC_asp + AC_low acceptance criteria

---

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install matplotlib  # also required but not yet in requirements.txt
```

---

## Running

```bash
# Quick smoke test
python negmasTest.py

# Compare agent evolution (v1 → v2 → v3)
python AgentEvolution.py

# Full benchmark against NegMAS built-in agents
python tournament_group31_agent.py

# Head-to-head vs Boulware opponent
python AdaptiveVSBoulware.py

# Head-to-head vs MiCRO opponent
python AdaptiveVSMicro.py
```
