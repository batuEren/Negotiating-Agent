
"""
micronegotiator.py
==================
A MiCRO-strategy SAO negotiator compatible with NegMAS.

MiCRO (Micro Concession Round-Robin) is a simple negotiation strategy
with no opponent modeling. It walks down the outcome space from best to
worst, one step at a time, proposing the next-best outcome on each turn.

Concession logic
----------------
Outcomes are pre-sorted by our utility function (highest first).
On turn 0 we propose the best outcome, on turn 1 the second-best, and so
on — like counting down: 1000, 999, 998, ...

We accept any incoming offer whose utility is >= our current proposal's
utility (i.e. "at least as good as what we are about to offer anyway").

Near the deadline (t >= 0.95) we fall back to accepting anything above
the effective reservation value.

Parameters
----------
reservation_ratio : float  (default 0.4)
    Fraction of the utility range used as the effective reservation value.
    Controls how far down we are willing to walk before the deadline.
"""

from negmas import Outcome, ResponseType
from negmas.sao import SAONegotiator, SAOState

class MicroNegotiator(SAONegotiator):
    """MiCRO: simple step-down negotiator for NegMAS SAO sessions."""

    def __init__(self, *args, reservation_ratio: float = 0.4, **kwargs):
        """Initialize the MicroNegotiator with reservation ratio parameter."""
        super().__init__(*args, **kwargs)
        self.reservation_ratio = reservation_ratio

        self._sorted_outcomes: list[Outcome] = []   # best -> worst
        self._step: int = 0                          # current position in list
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_ufun(self):
        """Get the active utility function (either ufun or preferences)."""
        ufun = getattr(self, "ufun", None)
        if ufun is None:
            ufun = getattr(self, "preferences", None)
        return ufun

    def _relative_time(self, state: SAOState | None = None) -> float:
        """Get the relative time (0 to 1) in the negotiation."""
        if state is not None:
            t = getattr(state, "relative_time", None)
            if t is not None:
                return float(t)
        if self.nmi is not None:
            t = getattr(self.nmi.state, "relative_time", None)
            if t is not None:
                return float(t)
        return 0.0

    def _utility_range(self):
        """Calculate the minimum acceptable utility (reservation value) and maximum utility."""
        ufun = self._active_ufun()
        if ufun is None:
            return 0.0, 0.0
        u_max = float(ufun.max())
        u_worst = float(ufun.min())
        u_min = u_worst + self.reservation_ratio * (u_max - u_worst)
        return u_min, u_max

    def _init_outcomes(self):
        """Sort all outcomes from best to worst utility (done once)."""
        if self._initialized or self.nmi is None:
            return
        ufun = self._active_ufun()
        if ufun is None:
            return

        self._sorted_outcomes = sorted(
            self.nmi.outcomes,
            key=lambda o: float(ufun(o)),
            reverse=True,   # best first
        )
        self._initialized = True

    def _current_offer(self) -> Outcome | None:
        """The outcome at our current step (does not advance the counter)."""
        if not self._sorted_outcomes:
            return None
        idx = min(self._step, len(self._sorted_outcomes) - 1)
        return self._sorted_outcomes[idx]

    # ------------------------------------------------------------------
    # SAO protocol callbacks
    # ------------------------------------------------------------------

    def propose(self, state: SAOState) -> Outcome | None:
        """Propose the next outcome in the pre-sorted list and advance the step counter."""
        self._init_outcomes()
        offer = self._current_offer()
        # Advance so the next call proposes the next-best outcome
        self._step += 1
        return offer

    def respond(self, state: SAOState) -> ResponseType:
        """Evaluate the opponent's offer and decide whether to accept or reject it."""
        self._init_outcomes()

        ufun = self._active_ufun()
        if ufun is None:
            return ResponseType.REJECT_OFFER

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offered_u = float(ufun(offer))

        # Accept if at least as good as our current proposal
        current = self._current_offer()
        if current is not None and offered_u >= float(ufun(current)):
            return ResponseType.ACCEPT_OFFER

        # Deadline safety net: accept anything above reservation
        u_min, _ = self._utility_range()
        t = self._relative_time(state)
        if t >= 0.95 and offered_u >= u_min:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER