"""
Causal Compression Model for Speaker Communication

Core from Kinney & Lombrozo (2024), "Building compressed causal models
of the world":
- Causal Mutual Information (CMI): Interventional mutual information
- Information Loss: L(C, C', E) = CMI(C, E) - CMI(C', E)

Extensions:
- RSA communication model: Speaker/listener over causal compressions
- Epistemic vigilance: Joint inference over (world complexity, speaker goal)
"""

# Kinney & Lombrozo (2024) framework
from .kinney_lombrozo import (
    Variable,
    CausalDAG,
    compute_cmi,
    compute_cmi_multivar,
    compute_information_loss,
    compress_dag,
)

# RSA communication model (our extension)
from .rsa import (
    Utterance,
    CompressionSpeaker,
    compute_contextual_kl,
    RSATrustModel,
    Goal,
    compute_rate_distortion_curve,
    compute_trust_curve,
)
