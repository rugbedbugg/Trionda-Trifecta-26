"""Full 2026 World Cup tournament simulator.

Predicts every match from the group stage to the final — applying the official
FIFA qualification rules to decide who advances from the groups — then validates
the predicted tournament against the results that actually happened.

Entry points:
    python -m wcpredictor.sim.simulate    # predict the whole tournament
    python -m wcpredictor.sim.validate     # score the prediction vs reality
"""
