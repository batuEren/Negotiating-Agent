"""
titForTatAgent.py
=================
A Tit-for-Tat SAO negotiator compatible with NegMAS.

Usage
-----
    from titForTatAgent import TitForTatAgent

    session.add(
        TitForTatAgent(name="buyer", reservation_ratio=0.4, alpha=1.0),
        ufun=buyer_utility,
    )

Parameters
----------
reservation_ratio : float  (default 0.4)
    Fraction of the utility range used as the effective reservation value.
    0.0 = accept anything, 1.0 = never concede below the best outcome.

alpha : float  (default 1.0)
    The matching factor for opponent concessions.
        alpha = 1.0 -> Exact Tit-for-Tat (matches concession exactly)
        alpha < 1.0 -> Hard-headed (concedes less than the opponent)
        alpha > 1.0 -> Accommodating (concedes more than the opponent)
"""

from negmas import Outcome, ResponseType
from negmas.sao import SAONegotiator, SAOState


class TitForTatAgent(SAONegotiator):
    """Tit-for-Tat negotiator for NegMAS SAO sessions."""

    def __init__(
        self,
        *args,
        reservation_ratio: float = 0.4,
        alpha: float = 1.0,
        opening_utility: float | None = None,
        **kwargs,
    ):
        """
        Initializes the negotiator.

        Args:
            *args:
            reservation_ratio:
            alpha:
            opening_utility:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.reservation_ratio = reservation_ratio
        self.alpha = alpha
        self.opening_utility = opening_utility
        
        # State tracking
        self._opponent_history = []
        self._my_current_target = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._opponent_history = []
        _, u_max = self._utility_range()
        if self.opening_utility is not None:
            self._my_current_target = self.opening_utility
        else:
            self._my_current_target = u_max

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_ufun(self):
        """Return the currently attached utility function."""
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

    def _best_offer_above(self, threshold: float) -> Outcome | None:
        """Return the outcome that is closest to (but not below) *threshold*."""
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

        return best_o if best_o is not None else ufun.best()

    # ------------------------------------------------------------------
    # SAO protocol callbacks
    # ------------------------------------------------------------------

    def respond(self, state: SAOState) -> ResponseType:
        """Evaluate the opponent's offer and track their concessions."""
        ufun = self._active_ufun()
        if ufun is None:
            return ResponseType.REJECT_OFFER

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offered_u = float(ufun(offer))
        self._opponent_history.append(offered_u)

        # Fallback if on_negotiation_start was skipped for some reason
        if self._my_current_target is None:
             self.on_negotiation_start(state)

        # Accept if the offer meets our current target
        if offered_u >= self._my_current_target:
            return ResponseType.ACCEPT_OFFER

        # Near the deadline: accept anything above the effective reservation
        u_min, _ = self._utility_range()
        t = float(getattr(state, "relative_time", 0.0))
        if t >= 0.95 and offered_u >= u_min:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def propose(self, state: SAOState) -> Outcome | None:
        """Propose an offer by matching the opponent's last concession."""
        u_min, u_max = self._utility_range()

        if self._my_current_target is None:
            self.on_negotiation_start(state)

        # Always concede a little bit to keep the negotiation moving
        self._my_current_target -= (u_max - u_min) / (self.nmi.n_steps * 2)

        # Calculate opponent's concession based on the utility of their offers to us
        if len(self._opponent_history) >= 2:
            latest_u = self._opponent_history[-1]
            previous_u = self._opponent_history[-2]
            
            # If positive, they made a better offer to us (they conceded)
            concession = latest_u - previous_u
            
            if concession > 0:
                # Lower our target utility by alpha * their concession
                self._my_current_target -= (self.alpha * concession)

        # Ensure our target stays within valid bounds
        self._my_current_target = max(u_min, min(u_max, self._my_current_target))

        return self._best_offer_above(self._my_current_target)