#from https://negmas.readthedocs.io/en/latest/tutorials/01.running_simple_negotiation.html#a-simple-bilateral-negotiation

from negmas import (
    make_issue,
    SAOMechanism,
    TimeBasedConcedingNegotiator,
)
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun
from titTat import TitForTatAgent
import matplotlib.pyplot as plt

# create negotiation agenda (issues)
issues = [
    make_issue(name="price", values=10),
    make_issue(name="quantity", values=(1, 11)),
    make_issue(name="delivery_time", values=10),
]
# create the mechanism
session = SAOMechanism(issues=issues, n_steps=20)
# define buyer and seller utilities
seller_utility = LUFun(
    values=[IdentityFun(), LinearFun(0.2), AffineFun(-1, bias=9.0)],
    outcome_space=session.outcome_space,
)
buyer_utility = LUFun(
    values={
        "price": AffineFun(-1, bias=9.0),
        "quantity": LinearFun(0.2),
        "delivery_time": IdentityFun(),
    },
    outcome_space=session.outcome_space,
)
# create and add buyer and seller negotiators
session.add(TitForTatAgent(name="buyer"), preferences=buyer_utility)
session.add(TimeBasedConcedingNegotiator(name="seller"), ufun=seller_utility)
# run the negotiation and show the results
session.run()
session.plot(show_reserved=False)
plt.show()