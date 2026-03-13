"""
Causal Compression Model for Speaker Communication

Core from Kinney & Lombrozo (2024), "Building compressed causal models
of the world":
- Causal Mutual Information (CMI): Interventional mutual information
- Information Loss: L(C, C', E) = CMI(C, E) - CMI(C', E)

Extensions:
- Context-dependent compression: Same G -> different optimal u for different c
- Speaker model: Softmax over contextual KL divergence loss
- RSA trust model: Joint inference over (world complexity, speaker goal)
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

# Speaker model (our extension)
from .speaker import Utterance, CompressionSpeaker, compute_contextual_kl

# RSA trust model (our extension; requires memo-lang + jax)
from .rsa import RSATrustModel, Goal

# Analysis tools (our extension)
from .analysis import (
    compute_rate_distortion_curve,
    compute_trust_curve,
)

# Example DAG builders
from .examples import (
    build_simple_medical_dag,
    build_complex_medical_dag,
    build_mask_advice_dag,
    build_drug_marker_scenario,
    build_trust_update_scenario,
)