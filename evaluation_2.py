from negmas import make_issue, SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun
import matplotlib.pyplot as plt
import plotly.express as px

from microNegotiator import MicroNegotiator
from adaptive_agent import AdaptiveNegotiator


def create_doc_style_scenario(n_steps=20):
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=(1, 11)),
        make_issue(name="delivery_time", values=10),
    ]
    session = SAOMechanism(issues=issues, n_steps=n_steps)

    seller_utility = LUFun(
        values={
            "price": IdentityFun(),
            "quantity": LinearFun(0.2),
            "delivery_time": AffineFun(-1, bias=9.0),
        },
        weights={"price": 1.0, "quantity": 1.0, "delivery_time": 10.0},
        outcome_space=session.outcome_space,
        reserved_value=15.0,
    ).scale_max(1.0)

    buyer_utility = LUFun(
        values={
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
        reserved_value=10.0,
    ).scale_max(1.0)

    return session, seller_utility, buyer_utility


def plot_trace_utilities(session):
    """
    Plot utilities of each offer in the extended trace.
    x-axis: offer index in the trace
    y-axis: utility of that offer for buyer/seller
    """
    if not session.extended_trace:
        print("No extended trace available.")
        return

    buyer = session.negotiators[0]
    seller = session.negotiators[1]

    offers = [offer for _, _, offer in session.extended_trace]
    buyer_utils = [buyer.ufun(o) for o in offers]
    seller_utils = [seller.ufun(o) for o in offers]

    x = list(range(len(offers)))

    plt.figure(figsize=(10, 5))
    plt.plot(x, buyer_utils, marker="o", label="Buyer utility of offers")
    plt.plot(x, seller_utils, marker="s", label="Seller utility of offers")
    plt.xlabel("Offer index in extended_trace")
    plt.ylabel("Utility")
    plt.title("Utilities of Offers Over Negotiation Trace")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def main():
    session, seller_utility, buyer_utility = create_doc_style_scenario(n_steps=20)

    session.add(MicroNegotiator(name="buyer"), ufun=buyer_utility)
    session.add(AdaptiveNegotiator(name="seller"), ufun=seller_utility)

    result = session.run()
    print(result)
    print(session.extended_trace)

    # 1) Built-in NegMAS plot
    session.plot(ylimits=(0.0, 1.01), show_reserved=True)

    # 2) Custom trace-utility plot
    plot_trace_utilities(session)
    plt.show()


if __name__ == "__main__":
    main()
