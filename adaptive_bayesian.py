from __future__ import annotations

import math
from itertools import permutations
from random import choice, sample, shuffle

from negmas import SAONegotiator, ResponseType, PreferencesChangeType
from negmas import PresortingInverseUtilityFunction

from negmas import make_issue, SAOMechanism, TimeBasedConcedingNegotiator
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun


class BayesianNegotiator(SAONegotiator):
    """
    Adaptive SAOP negotiator (v2 — Bayesian):
    - Bayesian learning opponent model   (Bayes' rule, Eq. 4-5)
    - Adaptive target with backstop      (slides 40, 45)
    - AC_asp + AC_low acceptance          (slide 64)
    """

    E = 3.0  # Boulware exponent for backstop curve
    BETA = 10.0  # Rationality parameter for Bayesian likelihood

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
        self._min_proposed_util = float("inf")

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
        """beta = max{ beta_0, beta_adapt }  —  never below reservation."""
        t = state.relative_time if state.relative_time is not None else 0.0
        beta = max(self._beta_0(t), self._beta_adapt(t))
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

        # Never accept below reservation value
        if offer_u < self._reservation:
            return ResponseType.REJECT_OFFER

        # AC_asp: accept if offer >= aspiration level
        if offer_u >= self._target(state):
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
            score = self._opp_util(o)
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
        return bid if bid is not None else self._best_outcome
