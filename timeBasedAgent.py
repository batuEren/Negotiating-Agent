"""
timeBasedAgent.py
=================
A simple time-based conceding SAO negotiator compatible with NegMAS.

Usage
-----
    from timeBasedAgent import TimeBasedAgent

    session.add(
        TimeBasedAgent(name="buyer", reservation_ratio=0.4, beta=1.0),
        ufun=buyer_utility,
    )

`preferences=...` is also supported for compatibility.

Parameters
----------
reservation_ratio : float  (default 0.4)
    Fraction of the utility range used as the effective reservation value.
    0.0 = accept anything, 1.0 = never concede below the best outcome.

beta : float  (default 1.0)
    Polynomial concession parameter (used only when concession_curve="poly"):
        beta > 1  ->  Boulware  (concedes slowly, stays firm until late)
        beta = 1  ->  Linear    (steady concession over time)
        beta < 1  ->  Conceder  (concedes quickly early on)

concession_curve : str  (default "poly")
    One of:
        "poly"        -> polynomial curve using beta
        "reverse_log" -> hard-headed early, faster concession late

reverse_log_k : float  (default 9.0)
    Shape parameter for "reverse_log". Larger values concede later.
"""

import math

from negmas import Outcome, ResponseType
from negmas.sao import SAONegotiator, SAOState


class TimeBasedAgent(SAONegotiator):
    """Time-based conceding negotiator for NegMAS SAO sessions."""

    def __init__(
        self,
        *args,
        reservation_ratio: float = 0.4,
        beta: float = 1.0,
        concession_curve: str = "poly",
        reverse_log_k: float = 9.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reservation_ratio = reservation_ratio
        self.beta = beta
        self.concession_curve = concession_curve
        self.reverse_log_k = reverse_log_k

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _relative_time(self, state: SAOState | None = None) -> float:
        """Normalised negotiation progress in [0, 1]."""
        if state is not None:
            t = getattr(state, "relative_time", None)
            if t is not None:
                return float(t)

        if self.nmi is not None:
            t = getattr(self.nmi.state, "relative_time", None)
            if t is not None:
                return float(t)

        return 0.0

    def _active_ufun(self):
        """
        Return the currently attached utility function.

        NegMAS can attach it under `ufun` (common) or `preferences`
        depending on API usage. This agent supports both.
        """
        ufun = getattr(self, "ufun", None)
        if ufun is None:
            ufun = getattr(self, "preferences", None)
        return ufun

    def _utility_range(self):
        """Return (u_min, u_max) based on the ufun and reservation_ratio."""
        ufun = self._active_ufun()
        if ufun is None:
            return 0.0, 0.0
        u_max = float(ufun.max())
        u_worst = float(ufun.min())
        u_min = u_worst + self.reservation_ratio * (u_max - u_worst)
        return u_min, u_max

    def _target_utility(self, t: float) -> float:
        """
        Time-based concession target utility.
        """
        u_min, u_max = self._utility_range()
        t = max(0.0, min(1.0, float(t)))

        if self.concession_curve == "reverse_log":
            # 0 at t=0 and 1 at t=1, with slow early growth.
            k = max(float(self.reverse_log_k), 1e-9)
            progress = 1.0 - math.log1p(k * (1.0 - t)) / math.log1p(k)
        else:
            # Polynomial fallback
            progress = t**self.beta

        return u_max - (u_max - u_min) * progress

    def _best_offer_above(self, threshold: float) -> Outcome | None:
        """
        Return the outcome that is closest to (but not below) *threshold*.
        This enforces concession over time instead of repeatedly proposing
        the global optimum.
        """
        if self.nmi is None:
            return None
        ufun = self._active_ufun()
        if ufun is None:
            return None

        best_gap, best_u, best_o = float("inf"), -1e9, None
        for outcome in self.nmi.outcomes:
            u = float(ufun(outcome))
            if u < threshold:
                continue
            gap = u - threshold
            if gap < best_gap - 1e-12 or (abs(gap - best_gap) <= 1e-12 and u > best_u):
                best_gap, best_u, best_o = gap, u, outcome

        # Fallback: propose the best available outcome rather than None
        return best_o if best_o is not None else ufun.best()

    # ------------------------------------------------------------------
    # SAO protocol callbacks
    # ------------------------------------------------------------------

    def propose(self, state: SAOState) -> Outcome | None:
        """Our turn to make an offer."""
        t = self._relative_time(state)
        target = self._target_utility(t)
        return self._best_offer_above(target)

    def respond(self, state: SAOState) -> ResponseType:
        """Evaluate the opponent's offer."""
        ufun = self._active_ufun()
        if ufun is None:
            return ResponseType.REJECT_OFFER

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offered_u = float(ufun(offer))
        t = self._relative_time(state)
        target = self._target_utility(t)

        # Accept if the offer meets our current aspiration level
        if offered_u >= target:
            return ResponseType.ACCEPT_OFFER

        # Near the deadline: accept anything above the effective reservation
        u_min, _ = self._utility_range()
        if t >= 0.95 and offered_u >= u_min:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
