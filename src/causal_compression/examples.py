"""
Example DAG builders for causal compression demonstrations.

All functions return pure data (no print statements).

Functions:
    build_simple_medical_dag: T -> Y (no moderators)
    build_complex_medical_dag: T -> Y <- M (with moderator)
    build_mask_advice_dag: R, M -> I (context-dependent mask effect)
    build_drug_marker_scenario: Returns (true_dag, utterances) tuple for demos
"""

import numpy as np
from typing import List, Tuple

from .dag import Variable, CausalDAG
from .speaker import Utterance


def build_simple_medical_dag() -> CausalDAG:
    """
    Simple medical DAG: Treatment -> Outcome

    No moderators or confounds.
    Treatment has a direct positive effect on outcome.
    """
    variables = {
        'T': Variable('T', (0, 1)),  # Treatment (0=no, 1=yes)
        'Y': Variable('Y', (0, 1)),  # Outcome (0=bad, 1=good)
    }

    parents = {
        'T': [],
        'Y': ['T'],
    }

    def cpt_T(parent_vals):
        return np.array([0.5, 0.5])  # Uniform prior on treatment

    def cpt_Y(parent_vals):
        t = parent_vals['T']
        if t == 1:
            return np.array([0.2, 0.8])  # Treatment effective
        else:
            return np.array([0.7, 0.3])  # No treatment worse

    cpts = {'T': cpt_T, 'Y': cpt_Y}

    return CausalDAG(variables, parents, cpts)


def build_complex_medical_dag() -> CausalDAG:
    """
    Complex medical DAG: Treatment effect moderated by context.

    T -> Y <- M

    M = moderator (e.g., disease severity, age group)
    Treatment effect depends on M.
    """
    variables = {
        'M': Variable('M', (0, 1)),  # Moderator (0=low risk, 1=high risk)
        'T': Variable('T', (0, 1)),  # Treatment
        'Y': Variable('Y', (0, 1)),  # Outcome
    }

    parents = {
        'M': [],
        'T': [],
        'Y': ['T', 'M'],
    }

    def cpt_M(parent_vals):
        return np.array([0.5, 0.5])  # Uniform

    def cpt_T(parent_vals):
        return np.array([0.5, 0.5])  # Uniform

    def cpt_Y(parent_vals):
        t = parent_vals['T']
        m = parent_vals['M']

        if m == 0:  # Low risk
            if t == 1:
                return np.array([0.1, 0.9])  # Treatment very effective
            else:
                return np.array([0.3, 0.7])  # Already okay without treatment
        else:  # High risk
            if t == 1:
                return np.array([0.4, 0.6])  # Treatment less effective
            else:
                return np.array([0.8, 0.2])  # Much worse without treatment

    cpts = {'M': cpt_M, 'T': cpt_T, 'Y': cpt_Y}

    return CausalDAG(variables, parents, cpts)


def build_mask_advice_dag(high_transmission: bool = False) -> CausalDAG:
    """
    Mask advice DAG: Effect varies by transmission rate.

    Variables:
    - R: Transmission rate (0=low, 1=high) - context
    - M: Mask wearing (0=no, 1=yes)
    - I: Infection (0=no, 1=yes) - outcome

    When R=low: masks have small effect
    When R=high: masks are crucial
    """
    variables = {
        'R': Variable('R', (0, 1)),
        'M': Variable('M', (0, 1)),
        'I': Variable('I', (0, 1)),
    }

    parents = {
        'R': [],
        'M': [],
        'I': ['R', 'M'],
    }

    def cpt_R(parent_vals):
        return np.array([0.5, 0.5])

    def cpt_M(parent_vals):
        return np.array([0.5, 0.5])

    def cpt_I(parent_vals):
        r = parent_vals['R']
        m = parent_vals['M']

        if r == 0:  # Low transmission
            if m == 1:
                return np.array([0.95, 0.05])  # Mask: almost no infection
            else:
                return np.array([0.90, 0.10])  # No mask: still low
        else:  # High transmission
            if m == 1:
                return np.array([0.70, 0.30])  # Mask: moderate protection
            else:
                return np.array([0.30, 0.70])  # No mask: high infection

    cpts = {'R': cpt_R, 'M': cpt_M, 'I': cpt_I}

    return CausalDAG(variables, parents, cpts)


def build_drug_marker_scenario() -> Tuple[CausalDAG, List[Utterance]]:
    """
    Build the drug/genetic marker scenario for context-dependent compression.

    Returns:
        Tuple of (true_dag, utterances) where:
        - true_dag: DAG with G (genetic marker) -> Y <- D (drug)
        - utterances: List of compressed utterances ["drug_works", "drug_doesnt_work", "full_model"]

    This is the canonical example for demonstrating context-dependent compression:
    - G=1 (has marker): Drug is highly effective (90% success)
    - G=0 (no marker): Drug is ineffective (20% success)
    """
    # Build true DAG: G -> Y <- D
    variables = {
        'G': Variable('G', (0, 1)),  # Genetic marker
        'D': Variable('D', (0, 1)),  # Drug taken
        'Y': Variable('Y', (0, 1)),  # Outcome (0=bad, 1=good)
    }
    parents = {'G': [], 'D': [], 'Y': ['G', 'D']}

    def cpt_G(p):
        return np.array([0.5, 0.5])

    def cpt_D(p):
        return np.array([0.5, 0.5])

    def cpt_Y(p):
        g, d = p['G'], p['D']
        if g == 1 and d == 1:
            return np.array([0.1, 0.9])   # Marker + Drug = great outcome
        elif g == 1 and d == 0:
            return np.array([0.4, 0.6])   # Marker + no drug = moderate
        elif g == 0 and d == 1:
            return np.array([0.8, 0.2])   # No marker + Drug = poor
        else:  # g == 0 and d == 0
            return np.array([0.6, 0.4])   # No marker + no drug = moderate

    true_dag = CausalDAG(variables, parents, {'G': cpt_G, 'D': cpt_D, 'Y': cpt_Y})

    # Build compressed utterances
    vars_simple = {'D': Variable('D', (0, 1)), 'Y': Variable('Y', (0, 1))}
    parents_simple = {'D': [], 'Y': ['D']}

    # Utterance: "Drug works"
    def cpt_Y_works(p):
        d = p['D']
        if d == 1:
            return np.array([0.1, 0.9])  # Drug works well
        else:
            return np.array([0.5, 0.5])  # Baseline without drug

    dag_works = CausalDAG(vars_simple, parents_simple, {'D': cpt_D, 'Y': cpt_Y_works})

    # Utterance: "Drug doesn't work"
    def cpt_Y_doesnt(p):
        d = p['D']
        if d == 1:
            return np.array([0.8, 0.2])  # Drug doesn't help
        else:
            return np.array([0.5, 0.5])  # Baseline

    dag_doesnt = CausalDAG(vars_simple, parents_simple, {'D': cpt_D, 'Y': cpt_Y_doesnt})

    utterances = [
        Utterance("drug_works", dag_works, "stability"),
        Utterance("drug_doesnt_work", dag_doesnt, "stability"),
        Utterance("full_model", true_dag, "full"),
    ]

    return true_dag, utterances
