from __future__ import annotations

import math
from itertools import permutations
from random import choice, gauss, sample, shuffle

from negmas import SAONegotiator, ResponseType, PreferencesChangeType
from negmas import PresortingInverseUtilityFunction

from negmas import make_issue, SAOMechanism, TimeBasedConcedingNegotiator
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun


class AdaptiveNegotiator(SAONegotiator):
    """
    Adaptive SAOP negotiator:
    - Bayesian learning opponent model   (Bayes' rule, Eq. 4-5)
    - PrONeg outcome prediction          (Florijn et al., 2026)
        Step 1: time-series forecasting of bid curves
        Step 2: Monte Carlo intersection → agreement probability
        Step 3: scenario integration → outcome ranking
    - Adaptive target with backstop      (slides 40, 45)
    - AC_asp + AC_low acceptance          (slide 64)
    """

    E = 3.0  # Boulware exponent for backstop curve
    BETA = 10.0  # Rationality parameter for Bayesian likelihood

    # PrONeg hyper-parameters
    MC_SAMPLES = 200  # Monte Carlo samples for intersection
    WINDOW = 5  # sliding-window size for time series (paper: 5-10)
    MIN_TS_POINTS = 5  # min data points before forecasting

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._inv = None
        self._min_util = None
        self._max_util = None
        self._best_outcome = None
        self._reservation = 0.0

        # Bayesian opponent model state
        self._issues = []
        self._issue_names = []
        self._w_hyps = []  # weight hypotheses (tuples on simplex)
        self._w_post = []  # weight posteriors
        self._v_hyps = {}  # {issue_name: list of {value: utility}}
        self._v_post = {}  # {issue_name: list of posteriors}

        self._opp_utils = []  # opponent offer utils (for us)
        self._own_utils = []  # our own bid utils (for us)

        # PrONeg state
        self._agree_prob = 0.5  # predicted agreement probability
        self._predicted_util = None  # predicted outcome utility (or None)

        # AC_low: track min utility we have proposed
        self._min_proposed_util = float("inf")

        self._util_cache = {}
        self._pool = []

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)

        changes = [c for c in changes if c.type not in (PreferencesChangeType.Scale,)]
        if not changes:
            return

        self._inv = PresortingInverseUtilityFunction(self.ufun)
        self._inv.init()

        worst, self._best_outcome = self.ufun.extreme_outcomes()
        self._best_outcome = tuple(self._best_outcome) if self._best_outcome else None
        worst = tuple(worst) if worst else None

        self._min_util = self.ufun(worst) if worst else 0.0
        self._max_util = self.ufun(self._best_outcome) if self._best_outcome else 1.0

        rv = self.ufun.reserved_value
        self._reservation = rv if rv is not None else self._min_util

        try:
            self._issues = list(self.nmi.issues)
            self._issue_names = [i.name for i in self._issues]
        except Exception:
            self._issues = []
            self._issue_names = []

        self._util_cache = {}
        self._opp_utils = []
        self._own_utils = []
        self._min_proposed_util = float("inf")
        self._agree_prob = 0.5
        self._predicted_util = None

        self._init_bayesian_model()
        self._build_pool()

    def _build_pool(self):
        self._pool = []
        try:
            raw = self._inv.some((self._min_util - 1e-6, self._max_util + 1e-6), False)
        except Exception:
            raw = []

        if not raw:
            self._pool = [self._best_outcome]
            return

        seen = set()
        unique = []
        for o in raw:
            o = tuple(o)
            if o not in seen:
                seen.add(o)
                unique.append(o)

        unique.sort(key=self._util, reverse=True)
        self._pool = unique[:500] if len(unique) > 500 else unique

        if not self._pool:
            self._pool = [self._best_outcome]

    # ------------------------------------------------------------------
    # Utility cache
    # ------------------------------------------------------------------
    def _util(self, offer):
        if offer is None:
            return float("-inf")
        offer = tuple(offer)
        if offer not in self._util_cache:
            self._util_cache[offer] = self.ufun(offer)
        return self._util_cache[offer]

    # ------------------------------------------------------------------
    # Bayesian opponent model  (Bayes' rule, Eq. 4-5)
    #
    # Hypotheses:
    #   - Weight hypotheses  H_w on discretised simplex
    #   - Value hypotheses   H_v per issue (permutations of utilities)
    #
    # Update  (each opponent offer = evidence E):
    #   P(H|E) = P(E|H) · P(H) / P(E)
    #   where P(E|H) ∝ exp(β · U_H(offer))
    #
    # Estimation:
    #   opp_util(o) = Σ E[w_i] · E[v_i(o_i)]
    # ------------------------------------------------------------------
    @staticmethod
    def _simplex_grid(n, step=0.1):
        """Generate weight vectors on the discretised simplex."""
        levels = int(round(1.0 / step))
        result = []

        def _recurse(remaining, dim, current):
            if dim == n - 1:
                current.append(remaining * step)
                result.append(tuple(current))
                current.pop()
                return
            for k in range(remaining + 1):
                current.append(k * step)
                _recurse(remaining - k, dim + 1, current)
                current.pop()

        _recurse(levels, 0, [])
        return result

    @staticmethod
    def _make_value_hyps(values, max_hyps=100):
        """Generate value-function hypotheses for one issue.

        Each hypothesis maps every value to a utility in [0, 1].
        Hypotheses are permutations of evenly-spaced utilities.
        """
        m = len(values)
        if m <= 1:
            return [{v: 1.0 for v in values}]

        utils = [i / (m - 1) for i in range(m)]

        if math.factorial(m) <= max_hyps:
            return [{values[i]: p[i] for i in range(m)} for p in permutations(utils)]

        # Too many permutations — sample a diverse subset
        seen = set()
        hyps = []
        asc = tuple(utils)
        desc = tuple(reversed(utils))
        for canonical in (asc, desc):
            seen.add(canonical)
            hyps.append({values[i]: canonical[i] for i in range(m)})
        while len(hyps) < max_hyps:
            perm = list(utils)
            shuffle(perm)
            key = tuple(perm)
            if key not in seen:
                seen.add(key)
                hyps.append({values[i]: perm[i] for i in range(m)})
        return hyps

    def _init_bayesian_model(self):
        """Initialise Bayesian hypotheses and uniform priors."""
        n = len(self._issues)
        if n == 0:
            self._w_hyps, self._w_post = [(1.0,)], [1.0]
            self._v_hyps, self._v_post = {}, {}
            return

        self._w_hyps = self._simplex_grid(n, step=0.1)
        self._w_post = [1.0 / len(self._w_hyps)] * len(self._w_hyps)

        self._v_hyps = {}
        self._v_post = {}
        for issue in self._issues:
            vals = list(issue.all)
            hyps = self._make_value_hyps(vals, max_hyps=100)
            self._v_hyps[issue.name] = hyps
            self._v_post[issue.name] = [1.0 / len(hyps)] * len(hyps)

    def _bayesian_update(self, offer):
        """Update all posteriors given one opponent offer (evidence E).

        Uses Bayes' rule (Eq. 5):
            P(H_i | E) = P(E | H_i) · P(H_i) / Σ_j P(E | H_j) · P(H_j)
        with likelihood  P(E | H) ∝ exp(β · U_H(offer)).
        """
        if offer is None:
            return
        offer = tuple(offer)

        # --- value posteriors (per issue, independent) ----------------
        for i, issue in enumerate(self._issues):
            if i >= len(offer):
                break
            name = issue.name
            val = offer[i]
            if name not in self._v_hyps:
                continue
            posteriors = self._v_post[name]
            hyps = self._v_hyps[name]
            new_post = [
                posteriors[h] * math.exp(self.BETA * hyp.get(val, 0.0))
                for h, hyp in enumerate(hyps)
            ]
            total = sum(new_post)
            if total > 0:
                self._v_post[name] = [p / total for p in new_post]

        # --- weight posteriors ----------------------------------------
        if self._w_hyps:
            new_w = []
            for h, w_hyp in enumerate(self._w_hyps):
                u = sum(
                    w_hyp[i] * self._expected_value(self._issues[i].name, offer[i])
                    for i in range(min(len(w_hyp), len(offer), len(self._issues)))
                )
                new_w.append(self._w_post[h] * math.exp(self.BETA * u))
            total = sum(new_w)
            if total > 0:
                self._w_post = [p / total for p in new_w]

    def _expected_value(self, issue_name, val):
        """E[v_j(val)] weighted by value-hypothesis posteriors."""
        if issue_name not in self._v_hyps:
            return 0.0
        return sum(
            p * h.get(val, 0.0)
            for p, h in zip(self._v_post[issue_name], self._v_hyps[issue_name])
        )

    def _expected_weights(self):
        """E[w] weighted by weight-hypothesis posteriors."""
        n = len(self._issues)
        weights = [0.0] * n
        for h, w_hyp in enumerate(self._w_hyps):
            p = self._w_post[h]
            for i in range(min(n, len(w_hyp))):
                weights[i] += p * w_hyp[i]
        return weights

    def _opp_util(self, offer):
        """Estimated opponent utility = Σ E[w_i] · E[v_i(o_i)]."""
        if offer is None or not self._issues:
            return 0.0
        offer = tuple(offer)
        weights = self._expected_weights()
        total = 0.0
        for i, issue in enumerate(self._issues):
            if i >= len(offer) or i >= len(weights):
                break
            total += weights[i] * self._expected_value(issue.name, offer[i])
        return total

    # ------------------------------------------------------------------
    # PrONeg outcome prediction  (Florijn et al., 2026, Sect. 3)
    #
    # Step 1: forecast both bid-utility curves with linear regression
    #         (windowed to reduce noise, as recommended by the paper)
    # Step 2: Monte Carlo intersection → agreement probability + utility
    #         distribution
    # Step 3: scenario integration → rank outcomes by predicted utility
    #         combined with the Bayesian opponent model
    # ------------------------------------------------------------------
    @staticmethod
    def _windowed(series, w):
        """Apply sliding-window max (opponent) or min (own) smoothing."""
        if w <= 1 or len(series) <= w:
            return list(series)
        return [max(series[max(0, i - w + 1) : i + 1]) for i in range(len(series))]

    @staticmethod
    def _windowed_min(series, w):
        """Sliding-window min (for own curve — agent concedes downward)."""
        if w <= 1 or len(series) <= w:
            return list(series)
        return [min(series[max(0, i - w + 1) : i + 1]) for i in range(len(series))]

    @staticmethod
    def _lin_regress(ys):
        """OLS regression y = a + b*x → (intercept, slope, residual_std).

        Lightweight substitute for Gaussian process regression.
        """
        n = len(ys)
        if n < 2:
            return (ys[0] if ys else 0.5), 0.0, 0.1
        xs = list(range(n))
        x_mean = (n - 1) / 2.0
        y_mean = sum(ys) / n
        ss_xx = sum((x - x_mean) ** 2 for x in xs)
        ss_xy = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        b = ss_xy / ss_xx if ss_xx > 0 else 0.0
        a = y_mean - b * x_mean
        residuals = [ys[i] - (a + b * xs[i]) for i in range(n)]
        var = sum(r * r for r in residuals) / n
        return a, b, max(math.sqrt(var), 0.01)

    def _forecast_curve(self, series, steps, use_min=False):
        """Step 1: forecast a bid-utility curve into the future.

        Returns list of (mean, std) for each future step.
        """
        w = self.WINDOW
        smoothed = (
            self._windowed_min(series, w) if use_min else self._windowed(series, w)
        )
        a, b, sigma = self._lin_regress(smoothed)
        n = len(smoothed)
        forecasts = []
        for s in range(1, steps + 1):
            t = n - 1 + s
            mu = a + b * t
            mu = max(0.0, min(1.0, mu))
            # Uncertainty grows with distance from observed data
            std = sigma * (1.0 + 0.1 * s)
            forecasts.append((mu, std))
        return forecasts

    def _run_proneg(self, state):
        """Run the PrONeg pipeline (Steps 1-2) and update predictions.

        Called after each opponent offer is recorded.
        """
        if len(self._opp_utils) < self.MIN_TS_POINTS:
            return  # not enough data yet

        t = state.relative_time if state.relative_time is not None else 0.0
        if t >= 1.0:
            return

        # How many future steps to forecast
        total_steps = state.step + 1 if state.step is not None else len(self._opp_utils)
        remaining = (
            max(5, int(total_steps / max(t, 0.01) - total_steps)) if t > 0 else 20
        )
        remaining = min(remaining, 200)

        # Step 1: forecast both curves
        opp_forecast = self._forecast_curve(self._opp_utils, remaining, use_min=False)
        own_forecast = self._forecast_curve(
            self._own_utils if self._own_utils else self._opp_utils,
            remaining,
            use_min=True,
        )

        # Step 2: Monte Carlo intersection
        agreements = 0
        agreement_utils = []
        for _ in range(self.MC_SAMPLES):
            prev_diff = None
            for k, ((omu, ostd), (amu, astd)) in enumerate(
                zip(opp_forecast, own_forecast)
            ):
                opp_sample = max(0.0, min(1.0, gauss(omu, ostd)))
                own_sample = max(0.0, min(1.0, gauss(amu, astd)))
                diff = own_sample - opp_sample
                if prev_diff is not None and prev_diff > 0 and diff <= 0:
                    # Intersection found: own curve dipped below opponent
                    util = (own_sample + opp_sample) / 2.0
                    agreements += 1
                    agreement_utils.append(util)
                    break
                prev_diff = diff

        self._agree_prob = agreements / self.MC_SAMPLES
        if agreement_utils:
            self._predicted_util = sum(agreement_utils) / len(agreement_utils)
        else:
            self._predicted_util = None

    def _outcome_score(self, offer):
        """Step 3: scenario integration — rank an outcome.

        Combines the Bayesian opponent model with the predicted outcome
        utility distribution.  Outcomes closer to the predicted utility
        and higher in opponent utility are ranked higher.
        """
        if offer is None:
            return float("-inf")
        own_u = self._util(offer)
        opp_u = self._opp_util(offer)

        # If we have a predicted outcome utility, favour offers close to it
        if self._predicted_util is not None:
            dist = abs(own_u - self._predicted_util)
            proximity = math.exp(-4.0 * dist)  # Gaussian-like kernel
        else:
            proximity = 1.0

        return opp_u * proximity

    # ------------------------------------------------------------------
    # Adaptive target with backstop  (slides 40, 45)
    #
    #   beta_0     = Boulware backstop (prevents exploitation)
    #   beta_adapt = adapted by opponent behaviour:
    #       opponent hardheaded  -> we concede  (lower target)
    #       opponent conceding   -> be hardheaded (raise target)
    #   beta = max{ beta_0, beta_adapt }
    # ------------------------------------------------------------------
    def _beta_0(self, t):
        """Backstop: time-based Boulware curve."""
        ratio = max(0.0, 1.0 - t**self.E)
        return self._min_util + ratio * (self._max_util - self._min_util)

    def _opponent_is_conceding(self):
        """True if opponent's recent offers are getting better for us."""
        if len(self._opp_utils) < 3:
            return None  # not enough data
        recent = self._opp_utils[-5:]
        return recent[-1] > recent[0]

    def _beta_adapt(self, t):
        """Adaptive target based on opponent behaviour."""
        ratio = max(0.0, 1.0 - t**self.E)

        conceding = self._opponent_is_conceding()
        if conceding is True:
            # Opponent conceding -> be hardheaded (raise target)
            ratio = min(1.0, ratio + 0.08)
        elif conceding is False:
            # Opponent hardheaded -> concede (lower target)
            ratio = max(0.0, ratio - 0.08)

        return self._min_util + ratio * (self._max_util - self._min_util)

    def _target(self, state):
        """beta = max{ beta_0, beta_adapt }, modulated by PrONeg.

        If PrONeg predicts low agreement probability, concede faster.
        If PrONeg has a predicted outcome utility, blend it in.
        Never below reservation.
        """
        t = state.relative_time if state.relative_time is not None else 0.0
        beta = max(self._beta_0(t), self._beta_adapt(t))

        # PrONeg modulation: agreement probability shapes urgency
        if self._agree_prob < 0.3 and t > 0.3:
            # Low agreement probability → concede to avoid breakoff
            beta = beta - 0.05 * (1.0 - self._agree_prob)
        elif self._agree_prob > 0.7:
            # High agreement probability → safe to hold firm
            beta = beta + 0.03 * self._agree_prob

        # If we have a predicted outcome utility, nudge target towards it
        if self._predicted_util is not None and t > 0.2:
            blend = min(0.3, t)  # trust prediction more as time passes
            beta = (1.0 - blend) * beta + blend * self._predicted_util

        return max(beta, self._reservation)

    # ------------------------------------------------------------------
    # Acceptance  (slide 64)
    #
    #   AC_asp: accept if  u(received) >= aspiration  lambda(t)
    #   AC_low: accept if  u(received) >= min{ u(w) | already proposed } U { u(w_next) }
    # ------------------------------------------------------------------
    def respond(self, state, source=None):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offer = tuple(offer)
        self._bayesian_update(offer)

        offer_u = self._util(offer)
        self._opp_utils.append(offer_u)

        # Run PrONeg prediction pipeline (Steps 1-2)
        self._run_proneg(state)

        # Never accept below reservation value
        if offer_u < self._reservation:
            return ResponseType.REJECT_OFFER

        # AC_asp: accept if offer >= aspiration level
        if offer_u >= self._target(state):
            return ResponseType.ACCEPT_OFFER

        # AC_proneg: if agreement looks unlikely and offer is reasonable,
        # accept to avoid breakoff  (paper Sect. 5: "Tactical Guidance")
        if (
            self._agree_prob < 0.25
            and offer_u >= self._reservation + 0.05
            and (state.relative_time or 0) > 0.7
        ):
            return ResponseType.ACCEPT_OFFER

        # AC_low: accept if offer >= min(our past proposals, our next proposal)
        next_bid = self._find_bid(state)
        next_u = self._util(next_bid) if next_bid is not None else float("inf")
        threshold = min(self._min_proposed_util, next_u)
        if offer_u >= threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    # ------------------------------------------------------------------
    # Bidding: from offers above target, pick the one opponent values most
    # ------------------------------------------------------------------
    def _find_bid(self, state):
        """Select bid using PrONeg Step 3: scenario integration.

        From offers above our target, rank by _outcome_score which
        combines the Bayesian opponent model with the predicted
        outcome utility distribution (Sect. 3.3 of the paper).
        """
        target = self._target(state)

        if not self._pool:
            return self._best_outcome

        above = [o for o in self._pool if self._util(o) >= target]

        if not above:
            return self._best_outcome

        candidates = sample(above, min(40, len(above)))

        best = None
        best_score = float("-inf")
        for o in candidates:
            score = self._outcome_score(o)
            if score > best_score:
                best_score = score
                best = o

        return best if best is not None else choice(candidates)

    def propose(self, state, dest=None):
        bid = self._find_bid(state)
        if bid is not None:
            u = self._util(bid)
            if u < self._min_proposed_util:
                self._min_proposed_util = u
            # Track own bid utilities for PrONeg time-series forecasting
            self._own_utils.append(u)
        return bid if bid is not None else self._best_outcome
