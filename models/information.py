"""
Information-theoretic functions for causal compression.

Functions:
    compute_cmi: Causal Mutual Information for single variable
    compute_cmi_multivar: CMI for multiple cause variables
    compute_information_loss: CMI-based information loss
    compute_context_conditioned_loss: KL divergence loss in context
    compute_voi: Value of Information
    compute_voli: Value of Lost Information
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from itertools import product

from .dag import CausalDAG


def compute_cmi(dag: CausalDAG, cause_var: str, effect_var: str) -> float:
    """
    Compute Causal Mutual Information CMI(C, E).

    CMI(C,E) = Σ_c q(c) Σ_e p(e|do(c)) log₂[p(e|do(c)) / p(e)]

    where q(c) is a reference distribution (uniform by default).

    This is INTERVENTIONAL mutual information, not correlational.
    """
    cause = dag.variables[cause_var]
    effect = dag.variables[effect_var]

    # Reference distribution q(c) - uniform
    q_c = 1.0 / len(cause.domain)

    # Compute p(e) = Σ_c q(c) p(e|do(c))
    p_e = {e: 0.0 for e in effect.domain}
    for c in cause.domain:
        joint_do_c = dag.compute_joint(interventions={cause_var: c})
        var_names = list(dag.variables.keys())
        e_idx = var_names.index(effect_var)
        for vals, prob in joint_do_c.items():
            p_e[vals[e_idx]] += q_c * prob

    # Compute CMI
    cmi = 0.0
    for c in cause.domain:
        joint_do_c = dag.compute_joint(interventions={cause_var: c})
        var_names = list(dag.variables.keys())
        e_idx = var_names.index(effect_var)

        # Get p(e|do(c))
        p_e_given_do_c = {e: 0.0 for e in effect.domain}
        for vals, prob in joint_do_c.items():
            p_e_given_do_c[vals[e_idx]] += prob

        for e in effect.domain:
            if p_e_given_do_c[e] > 0 and p_e[e] > 0:
                cmi += q_c * p_e_given_do_c[e] * np.log2(p_e_given_do_c[e] / p_e[e])

    return cmi


def compute_cmi_multivar(dag: CausalDAG, cause_vars: List[str], effect_var: str) -> float:
    """
    Compute CMI for multiple cause variables.

    CMI(C₁,...,Cₖ; E) with joint interventions.
    """
    effect = dag.variables[effect_var]

    # Enumerate all cause value combinations
    cause_domains = [dag.variables[c].domain for c in cause_vars]
    cause_combos = list(product(*cause_domains))

    # Uniform reference distribution
    q_c = 1.0 / len(cause_combos)

    # Compute marginal p(e)
    p_e = {e: 0.0 for e in effect.domain}
    var_names = list(dag.variables.keys())
    e_idx = var_names.index(effect_var)

    for combo in cause_combos:
        interventions = dict(zip(cause_vars, combo))
        joint = dag.compute_joint(interventions=interventions)
        for vals, prob in joint.items():
            p_e[vals[e_idx]] += q_c * prob

    # Compute CMI
    cmi = 0.0
    for combo in cause_combos:
        interventions = dict(zip(cause_vars, combo))
        joint = dag.compute_joint(interventions=interventions)

        p_e_given_do_c = {e: 0.0 for e in effect.domain}
        for vals, prob in joint.items():
            p_e_given_do_c[vals[e_idx]] += prob

        for e in effect.domain:
            if p_e_given_do_c[e] > 0 and p_e[e] > 0:
                cmi += q_c * p_e_given_do_c[e] * np.log2(p_e_given_do_c[e] / p_e[e])

    return cmi


def compute_information_loss(
    true_dag: CausalDAG,
    abstracted_dag: CausalDAG,
    cause_vars: List[str],
    effect_var: str
) -> float:
    """
    Compute information loss from compression.

    L(G, G̃) = CMI_G(C, E) - CMI_G̃(C, E)

    where CMI is computed using the respective DAG structures.
    """
    cmi_true = compute_cmi_multivar(true_dag, cause_vars, effect_var)
    cmi_abstracted = compute_cmi_multivar(abstracted_dag, cause_vars, effect_var)

    return max(0, cmi_true - cmi_abstracted)


def compute_context_conditioned_loss(
    true_dag: CausalDAG,
    abstracted_dag: CausalDAG,
    effect_var: str,
    context: Dict[str, int]
) -> float:
    """
    Compute context-conditioned information loss.

    L_c(G, G̃) = KL[ P_G(Y|c) || P_G̃(Y|c) ]

    This measures how much prediction accuracy is lost by using the
    abstracted model in a specific context.
    """
    effect = true_dag.variables[effect_var]

    # Compute P_G(Y|c) using true DAG with context interventions
    joint_true = true_dag.compute_joint(interventions=context)
    var_names_true = list(true_dag.variables.keys())
    e_idx_true = var_names_true.index(effect_var)

    p_y_true = {e: 0.0 for e in effect.domain}
    for vals, prob in joint_true.items():
        p_y_true[vals[e_idx_true]] += prob

    # Compute P_G̃(Y|c) using abstracted DAG
    # Handle case where context variables may be abstracted away
    context_filtered = {k: v for k, v in context.items()
                       if k in abstracted_dag.variables}

    joint_abstracted = abstracted_dag.compute_joint(interventions=context_filtered)
    var_names_abs = list(abstracted_dag.variables.keys())

    if effect_var not in var_names_abs:
        # Effect variable abstracted away - maximum loss
        return float('inf')

    e_idx_abs = var_names_abs.index(effect_var)

    p_y_abstracted = {e: 0.0 for e in effect.domain}
    for vals, prob in joint_abstracted.items():
        p_y_abstracted[vals[e_idx_abs]] += prob

    # Compute KL divergence
    kl = 0.0
    for e in effect.domain:
        if p_y_true[e] > 0:
            if p_y_abstracted[e] > 0:
                kl += p_y_true[e] * np.log2(p_y_true[e] / p_y_abstracted[e])
            else:
                kl += float('inf')  # Abstracted model assigns 0 to possible outcome

    return kl


def compute_voi(
    dag: CausalDAG,
    observation_var: str,
    decision_var: str,
    outcome_var: str,
    utility_fn: Optional[Callable] = None
) -> float:
    """
    Compute Value of Information (VOI) for observing a variable.

    VOI_u(O) = Σ_o P(o) max_a E[u|a;o] - max_a E[u|a]

    This is the expected gain in utility from observing O before deciding on action a.

    Args:
        dag: The causal DAG
        observation_var: Variable to observe (O)
        decision_var: Action/decision variable (a)
        outcome_var: Outcome variable that utility depends on (Y)
        utility_fn: Function mapping outcome values to utility (default: identity)

    Returns:
        VOI in utility units
    """
    if utility_fn is None:
        # Default: outcome value IS the utility
        utility_fn = lambda y: float(y)

    obs = dag.variables[observation_var]
    act = dag.variables[decision_var]
    out = dag.variables[outcome_var]

    var_names = list(dag.variables.keys())
    obs_idx = var_names.index(observation_var)
    out_idx = var_names.index(outcome_var)

    # Compute P(o) - marginal distribution of observation
    joint = dag.compute_joint()
    p_o = {o: 0.0 for o in obs.domain}
    for vals, prob in joint.items():
        p_o[vals[obs_idx]] += prob

    # Compute E[u|a] for each action (without observation)
    eu_by_action = {}
    for a in act.domain:
        # Intervene on action
        joint_do_a = dag.compute_joint(interventions={decision_var: a})
        # Compute expected utility
        eu = 0.0
        for vals, prob in joint_do_a.items():
            eu += prob * utility_fn(vals[out_idx])
        eu_by_action[a] = eu

    max_eu_no_obs = max(eu_by_action.values())

    # Compute E[u|a;o] for each action and observation
    eu_conditioned = {}  # (a, o) -> E[u|a,o]
    for o in obs.domain:
        for a in act.domain:
            # Intervene on action AND condition on observation
            joint_do_a = dag.compute_joint(interventions={decision_var: a})

            # Compute P(y|a,o) = P(y,o|do(a)) / P(o|do(a))
            p_y_o = {y: 0.0 for y in out.domain}
            p_o_given_a = 0.0

            for vals, prob in joint_do_a.items():
                if vals[obs_idx] == o:
                    p_o_given_a += prob
                    p_y_o[vals[out_idx]] += prob

            if p_o_given_a > 0:
                # Normalize to get P(y|a,o)
                p_y_o = {y: p / p_o_given_a for y, p in p_y_o.items()}
                eu_conditioned[(a, o)] = sum(p_y_o[y] * utility_fn(y) for y in out.domain)
            else:
                eu_conditioned[(a, o)] = 0.0

    # Compute VOI
    voi = 0.0
    for o in obs.domain:
        # Best action given observation o
        max_eu_given_o = max(eu_conditioned[(a, o)] for a in act.domain)
        voi += p_o[o] * max_eu_given_o

    voi -= max_eu_no_obs

    return max(0, voi)  # VOI is non-negative


def compute_voli(
    true_dag: CausalDAG,
    compressed_dag: CausalDAG,
    observation_var: str,
    decision_var: str,
    outcome_var: str,
    utility_fn: Optional[Callable] = None
) -> float:
    """
    Compute Value of Lost Information (VOLI) from compression.

    VOLI_u(O, Ô) = VOI_u(O) - VOI_u(Ô)

    This measures how much decision-relevant information is lost
    by using the compressed representation.

    Key result from Kinney & Lombrozo (2024):
    For prediction-accuracy utility, VOLI = Information Loss L

    Args:
        true_dag: The true (uncompressed) DAG
        compressed_dag: The compressed/abstracted DAG
        observation_var: Variable to observe
        decision_var: Action/decision variable
        outcome_var: Outcome variable
        utility_fn: Utility function

    Returns:
        VOLI in utility units
    """
    voi_true = compute_voi(true_dag, observation_var, decision_var, outcome_var, utility_fn)

    # For compressed DAG, observation var may be abstracted
    if observation_var in compressed_dag.variables:
        voi_compressed = compute_voi(compressed_dag, observation_var, decision_var,
                                     outcome_var, utility_fn)
    else:
        # Observation variable was abstracted away - VOI from it is 0
        voi_compressed = 0.0

    return voi_true - voi_compressed
