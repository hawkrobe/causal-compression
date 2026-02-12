"""
Speaker model for causal compression.

Classes:
    Utterance: A compressed/abstracted causal model utterance
    CompressionSpeaker: Speaker that trades off compression vs informativeness
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from scipy.special import softmax as scipy_softmax

from .dag import CausalDAG
from .information import compute_context_conditioned_loss


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
                loss = compute_context_conditioned_loss(
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
