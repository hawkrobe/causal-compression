"""
Kinney & Lombrozo (2024) causal compression framework.

Implements the formal framework from:
    Kinney, D., & Lombrozo, T. (2024). Building compressed causal models
    of the world. Cognitive Psychology, 155, 101682.

This module contains all and only the concepts defined in the paper:

    Variable / CausalDAG:
        Causal Bayesian network with interventional distributions p(e|do(c)).

    CMI(C, E):
        Causal Mutual Information (Section 1.7, p.6).
        CMI(C, E) = sum_c q(c) sum_e p(e|do(c)) log2[p(e|do(c)) / p(e)]

    compress_dag:
        Applies the compression formula (p.6):
        p(e|do(c')) = sum_{c in sigma^{-1}(c')} p(e|do(c)) * q(c) / q(c')

    compute_information_loss:
        L(C, C', E) = CMI(C, E) - CMI(C', E)  (p.7)

    compute_cmi_multivar:
        CMI for joint cause sets {C, B}, needed for stability measurement
        (Section 1.9, Appendix A p.31).

No extensions, speaker models, or trust models are defined here.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from itertools import product


# ---------------------------------------------------------------------------
# Core DAG representation
# ---------------------------------------------------------------------------

@dataclass
class Variable:
    """A variable in a causal DAG."""
    name: str
    domain: Tuple[int, ...]  # Possible values (e.g., (0, 1) for binary)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


@dataclass
class CausalDAG:
    """
    A causal DAG with conditional probability tables.

    Attributes:
        variables: Dict mapping variable names to Variable objects
        parents: Dict mapping variable names to list of parent names
        cpts: Dict mapping variable names to CPT functions
              CPT function signature: f(parent_values: dict) -> np.array of probs over domain
    """
    variables: Dict[str, Variable]
    parents: Dict[str, List[str]]
    cpts: Dict[str, Callable]

    def __post_init__(self):
        """Validate DAG structure."""
        # Check all parents exist
        for var, par_list in self.parents.items():
            for p in par_list:
                if p not in self.variables:
                    raise ValueError(f"Parent {p} of {var} not in variables")

        # Check for cycles (simple DFS)
        visited = set()
        rec_stack = set()

        def has_cycle(v):
            visited.add(v)
            rec_stack.add(v)
            for parent in self.parents.get(v, []):
                if parent not in visited:
                    if has_cycle(parent):
                        return True
                elif parent in rec_stack:
                    return True
            rec_stack.remove(v)
            return False

        for var in self.variables:
            if var not in visited:
                if has_cycle(var):
                    raise ValueError("DAG contains a cycle")

    def get_topological_order(self) -> List[str]:
        """Return variables in topological order (parents before children)."""
        in_degree = {v: 0 for v in self.variables}
        for var, pars in self.parents.items():
            in_degree[var] = len(pars)

        queue = [v for v, d in in_degree.items() if d == 0]
        order = []

        while queue:
            v = queue.pop(0)
            order.append(v)
            for var, pars in self.parents.items():
                if v in pars:
                    in_degree[var] -= 1
                    if in_degree[var] == 0:
                        queue.append(var)

        return order

    def sample(self, n: int = 1, interventions: Optional[Dict[str, int]] = None) -> List[Dict[str, int]]:
        """
        Sample from the DAG, optionally with interventions.

        NOTE: Not part of the paper's formal framework (which uses exact
        computation only). Provided as an implementation utility.
        """
        interventions = interventions or {}
        order = self.get_topological_order()
        samples = []

        for _ in range(n):
            sample = {}
            for var in order:
                if var in interventions:
                    sample[var] = interventions[var]
                else:
                    parent_vals = {p: sample[p] for p in self.parents.get(var, [])}
                    probs = self.cpts[var](parent_vals)
                    domain = self.variables[var].domain
                    sample[var] = np.random.choice(domain, p=probs)
            samples.append(sample)

        return samples

    def compute_joint(self, interventions: Optional[Dict[str, int]] = None) -> Dict[Tuple, float]:
        """
        Compute full joint distribution P(V) or P(V | do(X=x)).

        Returns dict mapping value tuples to probabilities.
        """
        interventions = interventions or {}
        order = self.get_topological_order()

        # Enumerate all possible value combinations
        var_names = list(self.variables.keys())
        domains = [self.variables[v].domain for v in var_names]

        joint = {}
        for vals in product(*domains):
            assignment = dict(zip(var_names, vals))

            # Check if assignment is consistent with interventions
            consistent = all(assignment[v] == interventions[v] for v in interventions)
            if not consistent:
                joint[vals] = 0.0
                continue

            # Compute probability
            prob = 1.0
            for var in order:
                if var in interventions:
                    continue  # Intervened variables don't contribute
                parent_vals = {p: assignment[p] for p in self.parents.get(var, [])}
                cpt_probs = self.cpts[var](parent_vals)
                domain = self.variables[var].domain
                val_idx = domain.index(assignment[var])
                prob *= cpt_probs[val_idx]

            joint[vals] = prob

        return joint


# ---------------------------------------------------------------------------
# CMI: Causal Mutual Information (Section 1.7, p.6)
# ---------------------------------------------------------------------------

def compute_cmi(dag: CausalDAG, cause_var: str, effect_var: str,
                q: Dict[int, float] = None) -> float:
    """
    Compute Causal Mutual Information CMI(C, E).

    CMI(C,E) = sum_c q(c) sum_e p(e|do(c)) log2[p(e|do(c)) / p(e)]

    where q(c) is a reference distribution over interventions on C (uniform
    by default), and p(e|do(c)) is the interventional distribution from the
    DAG.

    Args:
        dag: The causal DAG
        cause_var: Name of the cause variable
        effect_var: Name of the effect variable
        q: Optional reference distribution over cause values. Dict mapping
            each cause value to its weight. Defaults to uniform.
    """
    cause = dag.variables[cause_var]
    effect = dag.variables[effect_var]

    if q is None:
        q_weights = {c: 1.0 / len(cause.domain) for c in cause.domain}
    else:
        q_weights = dict(q)

    var_names = list(dag.variables.keys())
    e_idx = var_names.index(effect_var)

    # Compute p(e|do(c)) for each c, and accumulate p(e) = sum_c q(c) p(e|do(c))
    p_e = {e: 0.0 for e in effect.domain}
    p_e_given_do = {}

    for c in cause.domain:
        joint_do_c = dag.compute_joint(interventions={cause_var: c})
        p_ec = {e: 0.0 for e in effect.domain}
        for vals, prob in joint_do_c.items():
            p_ec[vals[e_idx]] += prob
        p_e_given_do[c] = p_ec
        for e in effect.domain:
            p_e[e] += q_weights[c] * p_ec[e]

    # CMI = sum_c q(c) sum_e p(e|do(c)) log2[p(e|do(c)) / p(e)]
    cmi = 0.0
    for c in cause.domain:
        for e in effect.domain:
            p_ec = p_e_given_do[c][e]
            if p_ec > 0 and p_e[e] > 0:
                cmi += q_weights[c] * p_ec * np.log2(p_ec / p_e[e])

    return cmi


def compute_cmi_multivar(dag: CausalDAG, cause_vars: List[str], effect_var: str) -> float:
    """
    Compute CMI for multiple cause variables with joint interventions.

    CMI(C1,...,Ck; E) using uniform q over all cause-value combinations.

    Used for stability measurement (Section 1.9, Appendix A p.31):
    the stability of C->E w.r.t. background B is measured by
    L = CMI({C,B}, E) - CMI(C, E).
    """
    effect = dag.variables[effect_var]
    cause_domains = [dag.variables[c].domain for c in cause_vars]
    cause_combos = list(product(*cause_domains))

    q_c = 1.0 / len(cause_combos)

    var_names = list(dag.variables.keys())
    e_idx = var_names.index(effect_var)

    # Compute p(e|do(c)) for each combo, and accumulate marginal p(e)
    p_e = {e: 0.0 for e in effect.domain}
    p_e_given_do = {}

    for combo in cause_combos:
        interventions = dict(zip(cause_vars, combo))
        joint = dag.compute_joint(interventions=interventions)
        p_ec = {e: 0.0 for e in effect.domain}
        for vals, prob in joint.items():
            p_ec[vals[e_idx]] += prob
        p_e_given_do[combo] = p_ec
        for e in effect.domain:
            p_e[e] += q_c * p_ec[e]

    # CMI
    cmi = 0.0
    for combo in cause_combos:
        for e in effect.domain:
            p_ec = p_e_given_do[combo][e]
            if p_ec > 0 and p_e[e] > 0:
                cmi += q_c * p_ec * np.log2(p_ec / p_e[e])

    return cmi


# ---------------------------------------------------------------------------
# Compression formula (p.6)
# ---------------------------------------------------------------------------

def compress_dag(
    dag: CausalDAG,
    cause_var: str,
    effect_var: str,
    compression_map: Dict[int, int],
    compressed_cause_name: str = None,
    q: Dict[int, float] = None,
) -> CausalDAG:
    """
    Apply the compression formula to produce a compressed DAG.

    Given a surjective compression function sigma (compression_map) from the
    range of cause variable C to a coarser set C', constructs a new DAG
    where the cause variable has the compressed domain and the effect CPT
    uses the averaged interventional distribution:

        p(e|do(c')) = sum_{c in sigma^{-1}(c')} p(e|do(c)) * q(c) / q(c')

    where q is a reference distribution over interventions on C (uniform
    by default).

    Args:
        dag: The detailed (uncompressed) DAG
        cause_var: Name of the cause variable to compress
        effect_var: Name of the effect variable
        compression_map: Dict mapping each original cause value to its
            compressed value. E.g., {0: 0, 1: 0, 2: 1, 3: 1} maps
            values {0,1} -> 0 and {2,3} -> 1.
        compressed_cause_name: Name for the compressed variable
            (defaults to cause_var)
        q: Optional reference distribution over cause values. Dict mapping
            each cause value to its weight. Defaults to uniform.

    Returns:
        A new CausalDAG with the compressed cause variable.
    """
    if compressed_cause_name is None:
        compressed_cause_name = cause_var

    cause = dag.variables[cause_var]
    effect = dag.variables[effect_var]

    # Determine compressed domain
    compressed_domain = tuple(sorted(set(compression_map.values())))

    # Build inverse map: compressed_value -> [original_values]
    inverse_map: Dict[int, List[int]] = {}
    for orig, comp in compression_map.items():
        inverse_map.setdefault(comp, []).append(orig)

    # Reference distribution q over original cause domain
    if q is None:
        q_weights = {c: 1.0 / len(cause.domain) for c in cause.domain}
    else:
        q_weights = dict(q)

    # Compute p(e|do(c')) for each compressed value
    var_names = list(dag.variables.keys())
    e_idx = var_names.index(effect_var)

    compressed_cpts: Dict[int, np.ndarray] = {}
    for c_hat in compressed_domain:
        originals = inverse_map[c_hat]
        q_c_hat = sum(q_weights[c] for c in originals)

        p_e_given_do_c_hat = np.zeros(len(effect.domain))
        for c in originals:
            joint = dag.compute_joint(interventions={cause_var: c})
            for vals, prob in joint.items():
                e_val = vals[e_idx]
                e_val_idx = effect.domain.index(e_val)
                p_e_given_do_c_hat[e_val_idx] += (q_weights[c] / q_c_hat) * prob

        compressed_cpts[c_hat] = p_e_given_do_c_hat

    # Build the compressed DAG
    variables = {
        compressed_cause_name: Variable(compressed_cause_name, compressed_domain),
        effect_var: Variable(effect_var, effect.domain),
    }
    parents = {compressed_cause_name: [], effect_var: [compressed_cause_name]}

    def cpt_cause(p):
        return np.ones(len(compressed_domain)) / len(compressed_domain)

    def cpt_effect(p):
        c_hat = p[compressed_cause_name]
        return compressed_cpts[c_hat]

    return CausalDAG(variables, parents, {
        compressed_cause_name: cpt_cause,
        effect_var: cpt_effect,
    })


# ---------------------------------------------------------------------------
# Information loss (p.7)
# ---------------------------------------------------------------------------

def compute_information_loss(
    dag_detailed: CausalDAG,
    cause_var_detailed: str,
    dag_compressed: CausalDAG,
    cause_var_compressed: str,
    effect_var: str,
    compression_map: Dict[int, int] = None,
) -> float:
    """
    Compute information loss from compression.

    L(C, C', E) = CMI(C, E) - CMI(C', E)

    where CMI(C, E) is computed in the detailed DAG and CMI(C', E) is
    computed in the compressed DAG. The cause variables may have different
    names and domain sizes in the two DAGs.

    When a compression_map is provided, the compressed CMI uses the induced
    reference distribution q(c') = sum_{c in sigma^{-1}(c')} q(c), where
    q(c) is uniform over the detailed cause domain. This is required for
    correct results when the compression groups have unequal sizes.
    """
    cmi_detailed = compute_cmi(dag_detailed, cause_var_detailed, effect_var)

    if compression_map is not None:
        # Compute induced q over compressed domain
        n_orig = len(dag_detailed.variables[cause_var_detailed].domain)
        q_orig = 1.0 / n_orig
        q_compressed: Dict[int, float] = {}
        for comp_val in compression_map.values():
            q_compressed[comp_val] = q_compressed.get(comp_val, 0.0) + q_orig
        cmi_compressed = compute_cmi(dag_compressed, cause_var_compressed,
                                     effect_var, q=q_compressed)
    else:
        cmi_compressed = compute_cmi(dag_compressed, cause_var_compressed,
                                     effect_var)

    return cmi_detailed - cmi_compressed
