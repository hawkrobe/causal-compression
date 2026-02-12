# Causal Compression Model for Speaker Communication

Extends Kinney & Lombrozo (2024) framework from single-agent representation to speaker-listener communication setting.

## Key Concepts

### From Kinney & Lombrozo (2024)
- **Causal Mutual Information (CMI)**: Interventional (not correlational) measure of causal influence
  - `CMI(C,E) = Σ q(c) p(e|do(c)) log₂[p(e|do(c)) / p(e)]`
- **Information Loss**: `L(G, G̃) = CMI_G(C,E) - CMI_G̃(C,E)`
- **Value of Information (VOI)**: Expected utility gain from observing variable before deciding
- **Value of Lost Information (VOLI)**: Decision-relevant information lost through compression

### Novel Extensions for Communication
- **Context-Conditioned Information Loss**: `L_c(G, G̃) = KL[P_G(Y|c) || P_G̃(Y|c)]`
- **Speaker Compression Model**: `P_S(u|G,c) ∝ exp[-α·L_c(G,G̃(u))]`
- **Context-Dependent Compression**: Same true model G → different optimal utterance u for different context c
- **Listener Complexity Prior**: Different priors P(C=complex) lead to different trust updates after revision

## Key Prediction

When a speaker revises advice (e.g., "drug works" → "drug doesn't work" for different patients):

| Listener Prior | Interpretation | Trust Update |
|----------------|----------------|--------------|
| P(simple) high | Inconsistency | Trust decreases |
| P(complex) high | Context-sensitivity | Trust maintained |

**Critical insight**: This is NOT motivated reasoning - it's rational Bayesian inference with different priors!

## Installation

```bash
pip install numpy scipy
# Optional for plotting:
pip install matplotlib
```

## Package Structure

```
causal_compression/
├── __init__.py              # Public API exports
├── dag.py                   # CausalDAG, Variable classes
├── information.py           # CMI, KL, information loss, VOI/VOLI
├── speaker.py               # CompressionSpeaker, Utterance
├── listener.py              # CompressionListener, TrustModel
├── analysis.py              # Rate-Distortion curve computation
├── examples.py              # Example DAG builders (medical, mask advice)
├── notebooks/
│   └── demonstrations.qmd   # Quarto notebook with interactive examples
├── test_causal_compression.py
└── README.md
```

## Usage

### Basic Import

```python
from causal_compression import (
    Variable, CausalDAG, Utterance,
    CompressionSpeaker, CompressionListener, TrustModel,
    compute_cmi, compute_context_conditioned_loss,
    compute_rate_distortion_curve
)

# Or import example scenarios
from causal_compression.examples import build_drug_marker_scenario
```

### Context-Dependent Compression Example

```python
from causal_compression import CompressionSpeaker
from causal_compression.examples import build_drug_marker_scenario

# Build scenario: drug effectiveness depends on genetic marker
true_dag, utterances = build_drug_marker_scenario()

# Create speaker model
speaker = CompressionSpeaker(
    true_dag=true_dag,
    utterances=utterances,
    effect_var='Y',
    alpha=10.0  # Rationality parameter
)

# Get context-dependent utterance probabilities
probs_g1 = speaker.get_utterance_probs({'G': 1})  # Patient has marker
probs_g0 = speaker.get_utterance_probs({'G': 0})  # Patient lacks marker

# Different contexts → different optimal utterances
optimal_g1 = speaker.get_optimal_utterance({'G': 1})  # "drug_works"
optimal_g0 = speaker.get_optimal_utterance({'G': 0})  # "drug_doesnt_work"
```

### Trust Dynamics Example

```python
from causal_compression import TrustModel

# Create listeners with different priors
simple_believer = TrustModel.create_simple_believer()  # P(simple) = 0.9
complex_believer = TrustModel.create_complex_believer()  # P(complex) = 0.9

# Same observation: advice revision
result_simple = simple_believer.update('revision')
result_complex = complex_believer.update('revision')

# Opposite trust updates!
print(result_simple['trust_delta'])   # Negative: trust decreased
print(result_complex['trust_delta'])  # Positive: trust increased
```

## Running Demonstrations

Render the Quarto notebook:

```bash
cd models/causal_compression/notebooks
quarto render demonstrations.qmd
```

Or run interactively in VS Code with the Quarto extension.

## Running Tests

```bash
cd models/causal_compression
pytest test_causal_compression.py -v
```

## Module Reference

### `dag.py` - Core DAG Representation
- `Variable`: A variable with name and domain
- `CausalDAG`: DAG with CPTs, sampling, joint computation, marginalization

### `information.py` - Information-Theoretic Functions
- `compute_cmi(dag, cause_var, effect_var)`: Single-variable CMI
- `compute_cmi_multivar(dag, cause_vars, effect_var)`: Multi-variable CMI
- `compute_information_loss(true_dag, abstracted_dag, cause_vars, effect_var)`: CMI-based loss
- `compute_context_conditioned_loss(true_dag, abstracted_dag, effect_var, context)`: KL loss
- `compute_voi(dag, obs_var, decision_var, outcome_var)`: Value of Information
- `compute_voli(true_dag, compressed_dag, ...)`: Value of Lost Information

### `speaker.py` - Speaker Model
- `Utterance`: Compressed model representation
- `CompressionSpeaker`: Chooses utterances by trading compression vs informativeness

### `listener.py` - Listener Models
- `CompressionListener`: Maintains beliefs about world complexity
- `TrustModel`: Joint inference over (world, reliability) for trust dynamics

### `analysis.py` - Analysis Tools
- `compute_rate_distortion_curve(dag, utterances, effect_var, contexts)`: RD trade-off
- `plot_rate_distortion_curve(rd_data)`: Visualization (requires matplotlib)

### `examples.py` - Example Scenarios
- `build_simple_medical_dag()`: T → Y
- `build_complex_medical_dag()`: T → Y ← M
- `build_mask_advice_dag()`: R, M → I
- `build_drug_marker_scenario()`: Returns (true_dag, utterances) tuple

## Connection to Rate-Distortion Theory

Following Zaslavsky et al. (2020), the speaker trades off:
- **Rate** (complexity): Entropy of utterance distribution
- **Distortion**: Expected information loss

The α parameter controls this trade-off:
- α → 0: Uniform distribution (low rate, high distortion)
- α → ∞: Deterministic optimal (high rate, low distortion)

## References

- Kinney, M., & Lombrozo, T. (2024). Causal Abstraction: A Theoretical Framework for Mechanistic Interpretability. *Cognition*, 248, 105790.
- Zaslavsky, N., Kemp, C., Regier, T., & Tishby, N. (2020). Efficient compression in color naming and its evolution. *PNAS*, 115(31), 7937-7942.
