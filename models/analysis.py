"""
Analysis tools for causal compression models.

Functions:
    compute_rate_distortion_curve: Compute the Rate-Distortion trade-off
    compute_trust_curve: Sweep P(C=complex) and compute trust_delta
"""

import numpy as np
from typing import Dict, List, Optional

from .dag import CausalDAG
from .speaker import Utterance, CompressionSpeaker


def compute_rate_distortion_curve(
    true_dag: CausalDAG,
    utterances: List[Utterance],
    effect_var: str,
    contexts: List[Dict[str, int]],
    alpha_range: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute the Rate-Distortion trade-off curve.

    Following Zaslavsky et al. (2020), the speaker trades off:
    - Rate (complexity): How much information the utterance conveys
    - Distortion: How much prediction accuracy is lost

    The alpha parameter controls this trade-off:
    - alpha -> 0: Uniform distribution (minimal complexity, maximal distortion)
    - alpha -> inf: Deterministic optimal (maximal complexity, minimal distortion)

    Args:
        true_dag: The true causal model
        utterances: Available utterances
        effect_var: Outcome variable
        contexts: List of contexts to average over
        alpha_range: Range of alpha values to compute

    Returns:
        Dict with 'alpha', 'rate', 'distortion' arrays
    """
    if alpha_range is None:
        alpha_range = np.logspace(-1, 2, 50)

    rates = []
    distortions = []

    for alpha in alpha_range:
        speaker = CompressionSpeaker(
            true_dag=true_dag,
            utterances=utterances,
            effect_var=effect_var,
            alpha=alpha
        )

        # Average over contexts
        avg_entropy = 0.0
        avg_distortion = 0.0

        for context in contexts:
            probs = speaker.get_utterance_probs(context)
            losses = speaker.compute_losses(context)

            # Rate = entropy of utterance distribution
            p_array = np.array(list(probs.values()))
            p_array = p_array[p_array > 1e-10]  # Filter out zeros
            entropy = -np.sum(p_array * np.log2(p_array))

            # Distortion = expected information loss
            expected_loss = sum(probs[u.name] * losses[u.name]
                               for u in utterances if losses[u.name] < float('inf'))

            avg_entropy += entropy
            avg_distortion += expected_loss

        avg_entropy /= len(contexts)
        avg_distortion /= len(contexts)

        rates.append(avg_entropy)
        distortions.append(avg_distortion)

    return {
        'alpha': alpha_range,
        'rate': np.array(rates),
        'distortion': np.array(distortions)
    }


def compute_trust_curve(
    world_dags: Dict[str, CausalDAG],
    utterances: List[Utterance],
    effect_var: str,
    observations: List[tuple],
    prior_complex_range: np.ndarray,
    prior_reliable: float = 0.8,
    speaker_alpha: float = 10.0,
    contexts: List[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Sweep P(C=complex) from 0 to 1 and compute trust_delta at each value.

    Uses RSATrustModel (memo-based) for each prior value.

    Args:
        world_dags: {'simple': dag, 'complex': dag}
        utterances: Shared 2-utterance set
        effect_var: Outcome variable name
        observations: List of (context_dict, utterance_name) pairs
        prior_complex_range: Array of P(C=complex) values to sweep
        prior_reliable: Prior P(speaker = reliable)
        speaker_alpha: Rationality parameter
        contexts: List of context dicts for speaker table precomputation

    Returns:
        Dict with 'prior_complex', 'trust_delta', 'complexity_delta' arrays
    """
    from .rsa import RSATrustModel

    trust_deltas = []
    complexity_deltas = []

    for p_complex in prior_complex_range:
        model = RSATrustModel(
            world_dags=world_dags,
            utterances=utterances,
            effect_var=effect_var,
            prior_world={'simple': 1.0 - p_complex, 'complex': float(p_complex)},
            prior_reliable=prior_reliable,
            speaker_alpha=speaker_alpha,
            contexts=contexts,
        )
        result = model.update(observations)
        trust_deltas.append(result['trust_delta'])
        complexity_deltas.append(result['complexity_delta'])

    return {
        'prior_complex': np.array(prior_complex_range),
        'trust_delta': np.array(trust_deltas),
        'complexity_delta': np.array(complexity_deltas),
    }
