"""
Core DAG representation for causal compression models.

Classes:
    Variable: A variable in a causal DAG
    CausalDAG: A causal DAG with conditional probability tables
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from itertools import product


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

        Args:
            n: Number of samples
            interventions: Dict of {var_name: value} for do() operations

        Returns:
            List of dictionaries mapping variable names to sampled values
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

        # Normalize for interventions
        total = sum(joint.values())
        if total > 0:
            joint = {k: v/total for k, v in joint.items()}

        return joint

    def marginalize(self, joint: Dict[Tuple, float], keep_vars: List[str]) -> Dict[Tuple, float]:
        """Marginalize joint distribution to keep only specified variables."""
        var_names = list(self.variables.keys())
        keep_indices = [var_names.index(v) for v in keep_vars]

        marginal = {}
        for vals, prob in joint.items():
            kept = tuple(vals[i] for i in keep_indices)
            marginal[kept] = marginal.get(kept, 0.0) + prob

        return marginal
