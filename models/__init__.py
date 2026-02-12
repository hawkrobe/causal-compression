"""
Causal Compression Model for Speaker Communication

Extends Kinney & Lombrozo (2024) framework from single-agent representation
to speaker-listener communication setting.

Key concepts:
- Causal Mutual Information (CMI): Interventional, not correlational
- Information Loss: L(G, G_tilde) = CMI(G,Y) - CMI(G_tilde,Y)
- Context-dependent compression: Same G -> different optimal u for different c
- Listener complexity prior: P(C) affects trust updates after revision
"""

# Core DAG representation
from .dag import Variable, CausalDAG

# Information-theoretic functions
from .information import (
    compute_cmi,
    compute_cmi_multivar,
    compute_information_loss,
    compute_context_conditioned_loss,
    compute_voi,
    compute_voli,
)

# Speaker model
from .speaker import Utterance, CompressionSpeaker

# Listener models
from .listener import CompressionListener, TrustModel

# RSA trust model (requires memo-lang + jax)
from .rsa import RSATrustModel

# Analysis tools
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

__all__ = [
    # Core
    'Variable',
    'CausalDAG',
    # Information
    'compute_cmi',
    'compute_cmi_multivar',
    'compute_information_loss',
    'compute_context_conditioned_loss',
    'compute_voi',
    'compute_voli',
    # Speaker
    'Utterance',
    'CompressionSpeaker',
    # Listener
    'CompressionListener',
    'TrustModel',
    'RSATrustModel',
    # Analysis
    'compute_rate_distortion_curve',
    'compute_trust_curve',
    # Examples
    'build_simple_medical_dag',
    'build_complex_medical_dag',
    'build_mask_advice_dag',
    'build_drug_marker_scenario',
    'build_trust_update_scenario',
]
