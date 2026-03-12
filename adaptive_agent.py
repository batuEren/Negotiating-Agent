from __future__ import annotations

from collections import defaultdict
from random import choice, sample

from negmas import SAONegotiator, ResponseType, PreferencesChangeType
from negmas import PresortingInverseUtilityFunction

from negmas import make_issue, SAOMechanism, TimeBasedConcedingNegotiator
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun


class AdaptiveNegotiator(SAONegotiator):
    """
    Adaptive SAOP negotiator:
    - Frequency Analysis opponent model  (slides 71-72)
    - Adaptive target with backstop      (slides 40, 45)
    - AC_asp + AC_low acceptance          (slide 64)
    """

    E = 3.0  # Boulware exponent for backstop curve

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._inv = None
        self._min_util = None
        self._max_util = None
        self._best_outcome = None
        self._reservation = 0.0

        # Frequency analysis state
        self._freq = {}  # {issue: {value: count}}
        self._total_offers = 0  # k — total opponent offers

        self._issue_names = []
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
            self._issue_names = [i.name for i in self.nmi.issues]
        except Exception:
            self._issue_names = []

        self._util_cache = {}
        self._freq = {}
        self._total_offers = 0
        self._opp_utils = []
        self._min_proposed_util = float("inf")

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
    # Frequency Analysis  (slides 71-72)
    #
    #   v_j(option)  = f(option) / k          — value evaluation
    #   w_j          = max{f(o)|o in I_j} / k — issue weight
    #   opp_util(o)  = sum( w_j * v_j(o_j) )
    # ------------------------------------------------------------------
    def _record_offer(self, offer):
        """Update frequency counts with a new opponent offer."""
        if offer is None:
            return
        offer = tuple(offer)
        self._total_offers += 1

        for i, val in enumerate(offer):
            name = self._issue_names[i] if i < len(self._issue_names) else f"i{i}"
            if name not in self._freq:
                self._freq[name] = defaultdict(int)
            self._freq[name][val] += 1

    def _value_eval(self, issue, value):
        """v_j(option) = f(option) / k"""
        if issue not in self._freq or self._total_offers == 0:
            return 0.0
        return self._freq[issue][value] / self._total_offers

    def _issue_weight(self, issue):
        """w_j = max{ f(o) | o in I_j } / k"""
        if issue not in self._freq or self._total_offers == 0:
            return 0.0
        return max(self._freq[issue].values()) / self._total_offers

    def _opp_util(self, offer):
        """Estimated opponent utility = sum( w_j * v_j(o_j) )"""
        if offer is None or not self._issue_names or self._total_offers == 0:
            return 0.0
        offer = tuple(offer)
        total = 0.0
        for i, val in enumerate(offer):
            if i >= len(self._issue_names):
                break
            name = self._issue_names[i]
            total += self._issue_weight(name) * self._value_eval(name, val)
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
        self._record_offer(offer)

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
