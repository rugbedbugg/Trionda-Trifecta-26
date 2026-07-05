"""`python -m wcpredictor.sim` — simulate the whole tournament, then validate it.

Trains the model once and shares it across both steps.
"""
from . import bracket, simulate, validate
from .predict import MatchPredictor


def main():
    pred = MatchPredictor()   # cross-era model, used by validate's match-level check
    b = bracket.derive()
    sim = simulate.simulate(b=b, report_stdout=True)   # from-scratch sim (both eras)
    validate.run(pred=pred, b=b, sim=sim)


if __name__ == "__main__":
    main()
