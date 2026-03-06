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

from typing import Optional
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
        self.opening_utility = float(opening_utility)
        self.concession_factor = float(concession_factor)
        self.my_last_proposal_utility = self.opening_utility
        self.opponent_offer_history: list[float] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_ufun(self):
        """
        Return the currently attached utility function (callable).
        NegMAS usually supplies this as `ufun` or `preferences`.
        """
        ufun = getattr(self, "ufun", None)
        if ufun is None:
            ufun = getattr(self, "preferences", None)
        return ufun

    def _best_offer_above(self, threshold: float) -> Optional[Outcome]:
        """
        Return the outcome that is closest to (but not below) *threshold*.
        If no outcome is >= threshold, return the best available outcome (highest utility).
        """
        if self.nmi is None:
            return None
        ufun = self._active_ufun()
        if ufun is None:
            return None

        best_gap = float("inf")
        best_u = -float("inf")
        best_o = None
        fallback_best_u = -float("inf")
        fallback_best_o = None

        # First pass: find outcomes >= threshold with smallest gap
        for outcome in self.nmi.outcomes:
            try:
                u = float(ufun(outcome))
            except Exception:
                # Skip outcomes that can't be evaluated
                continue

            # Track global best for fallback in the same pass
            if u > fallback_best_u:
                fallback_best_u = u
                fallback_best_o = outcome

            if u < threshold:
                continue
            gap = u - threshold
            # prefer smaller gap; if equal gap prefer higher utility
            if gap < best_gap - 1e-12 or (abs(gap - best_gap) <= 1e-12 and u > best_u):
                best_gap, best_u, best_o = gap, u, outcome

        # If found an outcome above threshold, return it
        if best_o is not None:
            return best_o

        # Otherwise, return the single best outcome by utility (fallback).
        return fallback_best_o

    def _get_target_utility(self, state: SAOState) -> float:
        """Calculates the target utility for the next proposal."""
        ufun = self._active_ufun()
        if ufun is None:
            return self.my_last_proposal_utility

        # On the first turn (or if second agent starts), propose the opening utility
        # `state.step` is typically 0 at start; keep the original conservative check.
        if state.step < 2:
            return self.opening_utility

        # Get opponent's most recent offer from the state
        opponent_offer = state.current_offer
        if opponent_offer is None:
            return self.my_last_proposal_utility  # defensive

        try:
            current_opponent_util = float(ufun(opponent_offer))
        except Exception:
            # If we can't evaluate the opponent offer, hold our previous utility
            return self.my_last_proposal_utility

        # If we have seen a previous offer from the opponent, calculate concession
        if self.opponent_offer_history:
            last_opponent_util = self.opponent_offer_history[-1]
            # concession > 0 means opponent moved to a better offer for *us*
            concession = current_opponent_util - last_opponent_util

            if concession > 0:
                # If opponent conceded (improved offer for us), mirror with concession_factor
                target = self.my_last_proposal_utility - concession * self.concession_factor
            else:
                # Opponent held firm or got tougher; keep our last utility
                target = self.my_last_proposal_utility
        else:
            # First time seeing an offer, hold firm (no baseline to compare)
            target = self.my_last_proposal_utility

        # Ensure sensible bounds:
        # - Respect a reservation value if the utility object exposes one.
        #   Different implementations may use 'reserved_value' or 'reservation_value'.
        reserved = getattr(ufun, "reserved_value", None)
        if reserved is None:
            reserved = getattr(ufun, "reservation_value", None)
        try:
            if reserved is not None:
                target = max(float(reserved), target)
        except Exception:
            pass

        # Don't propose above the opening utility
        target = min(self.opening_utility, target)

        return float(target)

    # ------------------------------------------------------------------
    # SAO protocol callbacks
    # ------------------------------------------------------------------

    def on_negotiation_start(self, state: SAOState):
        super().on_negotiation_start(state)
        self.my_last_proposal_utility = self.opening_utility
        self.opponent_offer_history = []

    def propose(self, state: SAOState) -> Optional[Outcome]:
        """Our turn to make an offer.

        Important: we compute our target based on the *current* opponent offer
        and the last recorded opponent offer. After computing our target we then
        append the current opponent utility to the history so that later turns
        will compare the *next* incoming opponent offer against this one.
        """
        target = self._get_target_utility(state)
        self.my_last_proposal_utility = float(target)

        # Record opponent's current offer AFTER calculating our move so the stored
        # history always represents the previous offers (needed for Tit-for-Tat).
        opponent_offer = state.current_offer
        if opponent_offer is not None:
            ufun = self._active_ufun()
            if ufun is not None:
                try:
                    self.opponent_offer_history.append(float(ufun(opponent_offer)))
                except Exception:
                    # ignore append if we can't evaluate
                    pass

        return self._best_offer_above(target)

    def respond(self, state: SAOState) -> ResponseType:
        """Evaluate the opponent's offer."""
        ufun = self._active_ufun()
        if ufun is None:
            return ResponseType.REJECT_OFFER

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        try:
            offered_u = float(ufun(offer))
        except Exception:
            return ResponseType.REJECT_OFFER

        # Calculate what we would propose next (target)
        next_target = self._get_target_utility(state)

        # Accept if the offer is at least as good as what we would propose next
        if offered_u >= next_target:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
