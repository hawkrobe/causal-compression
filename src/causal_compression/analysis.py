"""
Analysis tools for causal compression models.

Functions:
    compute_rate_distortion_curve: Compute the Rate-Distortion trade-off
    plot_rate_distortion_curve: Visualize the trade-off (requires matplotlib)
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


def plot_rate_distortion_curve(
    rd_data: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 4)
):
    """
    Plot the Rate-Distortion curve.

    Requires matplotlib (optional dependency).

    Args:
        rd_data: Dict from compute_rate_distortion_curve
        save_path: Optional path to save figure
        figsize: Figure size tuple (width, height)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Rate vs Distortion
    ax = axes[0]
    ax.plot(rd_data['rate'], rd_data['distortion'], 'b-', linewidth=2)
    ax.set_xlabel('Rate (bits)')
    ax.set_ylabel('Distortion (bits)')
    ax.set_title('Rate-Distortion Trade-off')
    ax.grid(True, alpha=0.3)

    # Rate vs Alpha
    ax = axes[1]
    ax.semilogx(rd_data['alpha'], rd_data['rate'], 'g-', linewidth=2)
    ax.set_xlabel(r'$\alpha$ (rationality)')
    ax.set_ylabel('Rate (bits)')
    ax.set_title('Rate vs Rationality')
    ax.grid(True, alpha=0.3)

    # Distortion vs Alpha
    ax = axes[2]
    ax.semilogx(rd_data['alpha'], rd_data['distortion'], 'r-', linewidth=2)
    ax.set_xlabel(r'$\alpha$ (rationality)')
    ax.set_ylabel('Distortion (bits)')
    ax.set_title('Distortion vs Rationality')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes
