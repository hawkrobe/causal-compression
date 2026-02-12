"""
Listener models for causal compression.

Classes:
    CompressionListener: Listener that maintains beliefs about world complexity
    TrustModel: Joint inference over (world, reliability) for trust dynamics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from .dag import CausalDAG
from .speaker import Utterance, CompressionSpeaker


class CompressionListener:
    """
    Listener model that maintains beliefs about world complexity.

    Key insight: Different complexity priors P(C) lead to different
    trust updates after observing advice revision.

    If P(C=complex) is high -> revision signals appropriate context-sensitivity
    If P(C=simple) is high -> revision signals unreliability
    """

    def __init__(
        self,
        possible_dags: Dict[str, CausalDAG],
        prior_complexity: Dict[str, float],
        speaker_model: Optional[CompressionSpeaker] = None
    ):
        """
        Args:
            possible_dags: Dict mapping DAG names to CausalDAG objects
            prior_complexity: Prior P(G) for each DAG name
            speaker_model: Optional speaker model for Bayesian inference
        """
        self.possible_dags = possible_dags
        self.beliefs = dict(prior_complexity)  # P(G)
        self.speaker_model = speaker_model
        self.observation_history: List[Tuple[Dict, str]] = []  # (context, utterance_name)

    def normalize_beliefs(self):
        """Normalize belief distribution."""
        total = sum(self.beliefs.values())
        if total > 0:
            self.beliefs = {k: v/total for k, v in self.beliefs.items()}

    def update_on_utterance(
        self,
        context: Dict[str, int],
        utterance: Utterance,
        speaker_alpha: float = 1.0
    ):
        """
        Update beliefs P(G) after observing speaker's utterance in context.

        P(G|u,c) proportional to P(u|G,c) * P(G)

        where P(u|G,c) comes from the speaker model.
        """
        self.observation_history.append((context, utterance.name))

        for dag_name, dag in self.possible_dags.items():
            # Create temporary speaker model for this DAG hypothesis
            temp_speaker = CompressionSpeaker(
                true_dag=dag,
                utterances=[utterance] + [u for u in (self.speaker_model.utterances
                                                       if self.speaker_model else [])
                           if u.name != utterance.name],
                effect_var=self.speaker_model.effect_var if self.speaker_model else list(dag.variables.keys())[-1],
                alpha=speaker_alpha
            )

            # Get P(u|G,c)
            probs = temp_speaker.get_utterance_probs(context)
            likelihood = probs.get(utterance.name, 1e-10)

            # Bayesian update
            self.beliefs[dag_name] *= likelihood

        self.normalize_beliefs()

    def update_on_revision(
        self,
        context1: Dict[str, int],
        utterance1: Utterance,
        context2: Dict[str, int],
        utterance2: Utterance,
        speaker_alpha: float = 1.0
    ) -> Dict[str, float]:
        """
        Update beliefs after observing advice revision.

        Returns the change in beliefs (delta) for analysis.
        """
        beliefs_before = dict(self.beliefs)

        # Update on first utterance
        self.update_on_utterance(context1, utterance1, speaker_alpha)

        # Update on second utterance (the revision)
        self.update_on_utterance(context2, utterance2, speaker_alpha)

        # Compute deltas
        deltas = {
            dag: self.beliefs[dag] - beliefs_before[dag]
            for dag in self.beliefs
        }

        return deltas

    def get_complexity_belief(self, complex_dag_names: List[str]) -> float:
        """Get total probability mass on complex DAGs."""
        return sum(self.beliefs[name] for name in complex_dag_names
                  if name in self.beliefs)

    def compute_trust_metric(self) -> float:
        """
        Compute a trust/reliability metric based on belief consistency.

        Higher values indicate the listener believes the speaker is
        appropriately adapting to context (rational), lower values
        indicate inconsistency/unreliability.
        """
        if len(self.observation_history) < 2:
            return 1.0  # No revision observed yet

        # Trust metric: weighted average of beliefs in "coherent" explanations
        # (DAGs that would produce the observed sequence of utterances)
        coherence_scores = []

        for dag_name, dag in self.possible_dags.items():
            # How well does this DAG explain the observation sequence?
            log_likelihood = 0.0
            for context, utt_name in self.observation_history:
                # Compute P(u|G,c) under this DAG
                temp_speaker = CompressionSpeaker(
                    true_dag=dag,
                    utterances=self.speaker_model.utterances if self.speaker_model else [],
                    effect_var=self.speaker_model.effect_var if self.speaker_model else list(dag.variables.keys())[-1],
                    alpha=1.0
                )
                probs = temp_speaker.get_utterance_probs(context)
                p = probs.get(utt_name, 1e-10)
                log_likelihood += np.log(p + 1e-10)

            coherence_scores.append((dag_name, log_likelihood, self.beliefs[dag_name]))

        # Trust = expected coherence under current beliefs
        trust = sum(score * belief for _, score, belief in coherence_scores)

        # Normalize to [0, 1]
        return 1.0 / (1.0 + np.exp(-trust))


class TrustModel:
    """
    Model for joint inference over (world complexity, speaker reliability).

    When a speaker revises advice, listeners update both their beliefs about:
    1. How complex the world is (simple vs complex)
    2. How reliable the speaker is

    Key insight: The same observation (advice revision) produces OPPOSITE
    trust updates depending on the listener's prior about world complexity.

    - If prior P(simple) is high: revision -> speaker unreliable
    - If prior P(complex) is high: revision -> speaker reliable (adapting appropriately)
    """

    def __init__(
        self,
        prior_world_simple: float = 0.5,
        prior_speaker_reliable: float = 0.8
    ):
        """
        Initialize trust model with independent priors.

        Args:
            prior_world_simple: P(world = simple)
            prior_speaker_reliable: P(speaker = reliable)
        """
        self.prior_world_simple = prior_world_simple
        self.prior_speaker_reliable = prior_speaker_reliable

        # Joint prior: P(world, speaker) assuming independence
        self._beliefs = {
            ('simple', 'reliable'): prior_world_simple * prior_speaker_reliable,
            ('simple', 'unreliable'): prior_world_simple * (1 - prior_speaker_reliable),
            ('complex', 'reliable'): (1 - prior_world_simple) * prior_speaker_reliable,
            ('complex', 'unreliable'): (1 - prior_world_simple) * (1 - prior_speaker_reliable),
        }

        # Default likelihoods for "revision" observation (advice changes)
        # P(observe revision | world, speaker)
        self._revision_likelihoods = {
            ('simple', 'reliable'): 0.01,    # Near-impossible: reliable speaker contradicting self in simple world
            ('simple', 'unreliable'): 0.25,  # Random advice gives inconsistency
            ('complex', 'reliable'): 0.81,   # Expected: adapting to context
            ('complex', 'unreliable'): 0.25, # Random advice
        }

    @property
    def beliefs(self) -> Dict[Tuple[str, str], float]:
        """Current joint beliefs P(world, speaker)."""
        return dict(self._beliefs)

    def set_likelihoods(self, likelihoods: Dict[Tuple[str, str], float]):
        """
        Set custom likelihoods for revision observation.

        Args:
            likelihoods: Dict mapping (world, speaker) tuples to P(revision | world, speaker)
        """
        self._revision_likelihoods = dict(likelihoods)

    def update(self, observation: str = 'revision') -> Dict[str, float]:
        """
        Update beliefs after observing speaker behavior.

        Args:
            observation: Type of observation ('revision' for advice change)

        Returns:
            Dict with 'trust_delta' (change in P(reliable)) and
            'complexity_delta' (change in P(complex))
        """
        if observation != 'revision':
            raise ValueError(f"Unknown observation type: {observation}")

        # Store prior marginals
        prior_reliable = self.get_reliability_belief()
        prior_complex = self.get_complexity_belief()

        # Bayesian update
        likelihoods = self._revision_likelihoods
        unnormalized = {k: self._beliefs[k] * likelihoods[k] for k in self._beliefs}
        total = sum(unnormalized.values())
        self._beliefs = {k: v / total for k, v in unnormalized.items()}

        # Compute deltas
        posterior_reliable = self.get_reliability_belief()
        posterior_complex = self.get_complexity_belief()

        return {
            'trust_delta': posterior_reliable - prior_reliable,
            'complexity_delta': posterior_complex - prior_complex,
            'prior_reliable': prior_reliable,
            'posterior_reliable': posterior_reliable,
            'prior_complex': prior_complex,
            'posterior_complex': posterior_complex,
        }

    def get_reliability_belief(self) -> float:
        """Get marginal P(speaker = reliable)."""
        return (self._beliefs[('simple', 'reliable')] +
                self._beliefs[('complex', 'reliable')])

    def get_complexity_belief(self) -> float:
        """Get marginal P(world = complex)."""
        return (self._beliefs[('complex', 'reliable')] +
                self._beliefs[('complex', 'unreliable')])

    def reset(self):
        """Reset beliefs to prior."""
        self._beliefs = {
            ('simple', 'reliable'): self.prior_world_simple * self.prior_speaker_reliable,
            ('simple', 'unreliable'): self.prior_world_simple * (1 - self.prior_speaker_reliable),
            ('complex', 'reliable'): (1 - self.prior_world_simple) * self.prior_speaker_reliable,
            ('complex', 'unreliable'): (1 - self.prior_world_simple) * (1 - self.prior_speaker_reliable),
        }

    @classmethod
    def create_simple_believer(cls, reliability_prior: float = 0.8) -> 'TrustModel':
        """Create a listener who believes the world is simple (P(simple) = 0.9)."""
        return cls(prior_world_simple=0.9, prior_speaker_reliable=reliability_prior)

    @classmethod
    def create_complex_believer(cls, reliability_prior: float = 0.8) -> 'TrustModel':
        """Create a listener who believes the world is complex (P(complex) = 0.9)."""
        return cls(prior_world_simple=0.1, prior_speaker_reliable=reliability_prior)
