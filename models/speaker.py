"""
Speaker model for causal compression.

Classes:
    Utterance: A compressed/abstracted causal model utterance
    CompressionSpeaker: Speaker that trades off compression vs informativeness

Functions:
    compute_contextual_kl: KL divergence between true and compressed
        predictions in a specific context. This is NOT the global
        information loss L -- it is a context-specific prediction loss
        used by the speaker model.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from scipy.special import softmax as scipy_softmax

from .kinney_lombrozo import CausalDAG


# ---------------------------------------------------------------------------
# Context-specific KL divergence (our extension, not in the paper)
# ---------------------------------------------------------------------------

def compute_contextual_kl(
    true_dag: CausalDAG,
    abstracted_dag: CausalDAG,
    effect_var: str,
    context: Dict[str, int]
) -> float:
    """
    KL divergence between true and compressed effect predictions in context.

    KL[ P_G(Y|do(c)) || P_compressed(Y|do(c)) ]

    This measures how much prediction accuracy is lost by using the
    compressed model in a specific context c. It is used as the speaker's
    loss function.

    This is NOT the global information loss L(C, C', E) (which is a CMI
    difference averaged over all interventions). This is a context-specific
    measure for the speaker model.
    """
    effect = true_dag.variables[effect_var]

    # P_G(Y | do(c))
    joint_true = true_dag.compute_joint(interventions=context)
    var_names_true = list(true_dag.variables.keys())
    e_idx_true = var_names_true.index(effect_var)

    p_y_true = {e: 0.0 for e in effect.domain}
    for vals, prob in joint_true.items():
        p_y_true[vals[e_idx_true]] += prob

    # P_compressed(Y | do(c)), filtering context to variables in compressed DAG
    context_filtered = {k: v for k, v in context.items()
                       if k in abstracted_dag.variables}

    joint_abs = abstracted_dag.compute_joint(interventions=context_filtered)
    var_names_abs = list(abstracted_dag.variables.keys())

    if effect_var not in var_names_abs:
        return float('inf')

    e_idx_abs = var_names_abs.index(effect_var)

    p_y_abs = {e: 0.0 for e in effect.domain}
    for vals, prob in joint_abs.items():
        p_y_abs[vals[e_idx_abs]] += prob

    # KL(P_true || P_compressed)
    kl = 0.0
    for e in effect.domain:
        if p_y_true[e] > 0:
            if p_y_abs[e] > 0:
                kl += p_y_true[e] * np.log2(p_y_true[e] / p_y_abs[e])
            else:
                return float('inf')

    return kl


# ---------------------------------------------------------------------------
# Utterance
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    """
    An utterance representing a compressed/abstracted causal model.

    Attributes:
        name: Human-readable description
        abstracted_dag: The simplified DAG this utterance implies
        compression_type: 'proportionality' (coarsening) or 'stability' (eliding)
    """
    name: str
    abstracted_dag: CausalDAG
    compression_type: str  # 'proportionality', 'stability', or 'full'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


# ---------------------------------------------------------------------------
# CompressionSpeaker
# ---------------------------------------------------------------------------

class CompressionSpeaker:
    """
    Speaker model that chooses utterances by trading off compression vs informativeness.

    P_S(u|G,c) proportional to exp[-alpha * L_c(G, G_tilde(u))] * I[Valid(G_tilde(u), G)]

    Key property: Same G can produce different optimal u for different c
    (context-dependent compression).
    """

    def __init__(
        self,
        true_dag: CausalDAG,
        utterances: List[Utterance],
        effect_var: str,
        alpha: float = 1.0,
        validity_check: Optional[Callable] = None
    ):
        """
        Args:
            true_dag: The true causal model G
            utterances: Available compressed descriptions
            effect_var: The outcome variable of interest
            alpha: Rationality parameter (higher = more optimal)
            validity_check: Optional function (true_dag, utterance) -> bool
        """
        self.true_dag = true_dag
        self.utterances = utterances
        self.effect_var = effect_var
        self.alpha = alpha
        self.validity_check = validity_check or (lambda g, u: True)

    def compute_losses(self, context: Dict[str, int]) -> Dict[str, float]:
        """Compute information loss for each utterance given context."""
        losses = {}
        for u in self.utterances:
            if self.validity_check(self.true_dag, u):
                loss = compute_contextual_kl(
                    self.true_dag,
                    u.abstracted_dag,
                    self.effect_var,
                    context
                )
                losses[u.name] = loss
            else:
                losses[u.name] = float('inf')
        return losses

    def get_utterance_probs(self, context: Dict[str, int]) -> Dict[str, float]:
        """
        Compute P_S(u|G,c) for all utterances.

        Returns dict mapping utterance names to probabilities.
        """
        losses = self.compute_losses(context)

        # Filter out invalid utterances (inf loss)
        valid_losses = {u: l for u, l in losses.items() if l < float('inf')}

        if not valid_losses:
            # No valid utterances - uniform over all
            n = len(self.utterances)
            return {u.name: 1.0/n for u in self.utterances}

        # Softmax over negative losses
        names = list(valid_losses.keys())
        neg_losses = np.array([-valid_losses[n] for n in names])
        probs = scipy_softmax(self.alpha * neg_losses)

        result = {u.name: 0.0 for u in self.utterances}
        for name, prob in zip(names, probs):
            result[name] = prob

        return result

    def sample_utterance(self, context: Dict[str, int]) -> Utterance:
        """Sample an utterance given context."""
        probs = self.get_utterance_probs(context)
        names = list(probs.keys())
        p = [probs[n] for n in names]

        chosen_name = np.random.choice(names, p=p)
        return next(u for u in self.utterances if u.name == chosen_name)

    def get_optimal_utterance(self, context: Dict[str, int]) -> Utterance:
        """Get the most probable utterance given context."""
        probs = self.get_utterance_probs(context)
        best_name = max(probs, key=probs.get)
        return next(u for u in self.utterances if u.name == best_name)
