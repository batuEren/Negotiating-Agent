"""
titTat.py
=========
A Tit-for-Tat SAO negotiator compatible with NegMAS, inspired by the design
of the TimeBasedAgent.

Usage
-----
    from titTat import TitForTatAgent

    session.add(
        TitForTatAgent(name="buyer", opening_utility=0.95),
        ufun=buyer_utility,
    )

Parameters
----------
opening_utility : float (default 0.95)
    The utility of the first offer the agent will make.

concession_factor : float (default 1.0)
    A factor to control the amount of concession.
    - 1.0: Standard Tit-for-Tat (concedes the same amount as the opponent).
    - > 1.0: More concessive.
    - < 1.0: Less concessive.
"""

from negmas import Outcome, ResponseType
from negmas.sao import SAONegotiator, SAOState


class TitForTatAgent(SAONegotiator):
    """A Tit-for-Tat negotiator for NegMAS SAO sessions."""

    def __init__(
        self,
        *args,
        opening_utility: float = 0.95,
        concession_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.opening_utility = opening_utility
        self.concession_factor = concession_factor
        self.my_last_utility = self.opening_utility
        self.opponent_last_utility = 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _best_offer_above(self, threshold: float) -> Outcome | None:
        """
        Return the outcome that is closest to (but not below) *threshold*.
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

    def _target_utility(self, state: SAOState) -> float:
        """Calculates the target utility for the next proposal."""
        if state.step == 0:
            return self.opening_utility

        ufun = self._active_ufun()
        if ufun is None:
            return self.my_last_utility

        # Get the opponent's last offer and its utility for us
        opponent_offer = state.current_offer
        if opponent_offer is None:
            return self.my_last_utility

        current_opponent_utility_for_us = float(ufun(opponent_offer))

        # We don't know the opponent's utility function, so we estimate their
        # concession by looking at how the utility *for us* changes.
        opponent_concession = current_opponent_utility_for_us - self.opponent_last_utility
        self.opponent_last_utility = current_opponent_utility_for_us

        # Concede by a similar amount
        target = self.my_last_utility + opponent_concession * self.concession_factor
        
        # Ensure we don't go above our opening utility or below reservation
        target = min(self.opening_utility, target)
        if ufun.reserved_value is not None:
             target = max(target, ufun.reserved_value)

        return target

    # ------------------------------------------------------------------
    # SAO protocol callbacks
    # ------------------------------------------------------------------

    def on_negotiation_start(self, state: SAOState):
        super().on_negotiation_start(state)
        self.my_last_utility = self.opening_utility
        self.opponent_last_utility = 0.0 # Assume opponent starts with their worst offer for us

    def propose(self, state: SAOState) -> Outcome | None:
        """Our turn to make an offer."""
        target = self._target_utility(state)
        self.my_last_utility = target
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

        # Calculate what we would propose next
        # We need to simulate one step ahead for the target utility
        next_target = self._target_utility(state)

        # Accept if the offer is better than what we would propose
        if offered_u >= next_target:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
