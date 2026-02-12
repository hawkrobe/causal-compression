# Causal Compression

Extends the causal abstraction framework of Kinney & Lombrozo (2024) from single-agent representation to speaker-listener communication. Speakers choose compressed utterances that trade off complexity against context-conditioned information loss; listeners perform joint inference over world complexity and speaker reliability.

## Installation

```bash
pip install -e .
```

## Quick start

```python
from causal_compression import CompressionSpeaker, TrustModel
from causal_compression.examples import build_drug_marker_scenario

# Drug effectiveness depends on a genetic marker
true_dag, utterances = build_drug_marker_scenario()
speaker = CompressionSpeaker(true_dag=true_dag, utterances=utterances, effect_var='Y', alpha=10.0)

# Same true DAG, different optimal utterance per context
speaker.get_optimal_utterance({'G': 1})  # "drug_works"
speaker.get_optimal_utterance({'G': 0})  # "drug_doesnt_work"

# Listeners with different complexity priors update trust in opposite directions
simple = TrustModel.create_simple_believer(reliability_prior=0.8)
complex = TrustModel.create_complex_believer(reliability_prior=0.8)
simple.update('revision')['trust_delta']   # negative
complex.update('revision')['trust_delta']  # positive
```

See `notebooks/demonstrations.qmd` for worked examples with visualizations.

## References

- Kinney, M., & Lombrozo, T. (2024). Causal Abstraction: A Theoretical Framework for Mechanistic Interpretability. *Cognition*, 248, 105790.
- Zaslavsky, N., Kemp, C., Regier, T., & Tishby, N. (2020). Efficient compression in color naming and its evolution. *PNAS*, 115(31), 7937-7942.
