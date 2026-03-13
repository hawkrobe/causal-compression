"""
RSA communication model for causal compression.

Builds on Kinney & Lombrozo (2024) causal compression framework
and the epistemic vigilance / selective truth-telling framework.

Components:
    Utterance: A compressed causal model (the speaker's message)
    compute_contextual_kl: Context-specific prediction loss (KL divergence)
    CompressionSpeaker: Speaker that trades off compression vs informativeness
    RSATrustModel: Vigilant listener that jointly infers world and speaker goal
    compute_rate_distortion_curve: Rate-distortion trade-off analysis
    compute_trust_curve: Trust delta sweep over priors
"""

import numpy as np
import jax.numpy as jnp
from memo import memo
from enum import IntEnum
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.special import softmax as scipy_softmax

from .kinney_lombrozo import CausalDAG


# ---------------------------------------------------------------------------
# Context-specific KL divergence
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

    # P_G(Y | do(c)), filtering context to variables in true DAG
    context_true = {k: v for k, v in context.items()
                    if k in true_dag.variables}
    joint_true = true_dag.compute_joint(interventions=context_true)
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
        self.true_dag = true_dag
        self.utterances = utterances
        self.effect_var = effect_var
        self.alpha = alpha
        self.validity_check = validity_check or (lambda g, u: True)

    def compute_losses(self, context: Dict[str, int]) -> Dict[str, float]:
        """Compute contextual KL loss for each utterance given context."""
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


# ---------------------------------------------------------------------------
# RSA domains (IntEnum, following the memo pattern)
# ---------------------------------------------------------------------------

class Utt(IntEnum):
    """Two-utterance domain for trust scenarios."""
    U0 = 0
    U1 = 1


class Goal(IntEnum):
    """Speaker goal types.

    - INFORMATIVE: minimize information loss (standard RSA)
    - PERSUADE_UP: maximize listener's belief that outcome Y=1
    - PERSUADE_DOWN: maximize listener's belief that outcome Y=0
    - UNRELIABLE: uniform random over utterances (incompetent/noisy)
    """
    INFORMATIVE = 0
    PERSUADE_UP = 1
    PERSUADE_DOWN = 2
    UNRELIABLE = 3


N_GOALS = len(Goal)


class WG(IntEnum):
    """Joint state: (world_type, goal) packed into a single index.
       w: 0 = simple, 1 = complex
       g: 0 = informative, 1 = persuade_up, 2 = persuade_down, 3 = unreliable
       Packing: s = w * N_GOALS + g  ->  w = s // N_GOALS,  g = s % N_GOALS
    """
    SIMPLE_INFORMATIVE = 0
    SIMPLE_PERSUADE_UP = 1
    SIMPLE_PERSUADE_DOWN = 2
    SIMPLE_UNRELIABLE = 3
    COMPLEX_INFORMATIVE = 4
    COMPLEX_PERSUADE_UP = 5
    COMPLEX_PERSUADE_DOWN = 6
    COMPLEX_UNRELIABLE = 7


N_STATES = len(WG)

GOAL_NAMES = ['informative', 'persuade_up', 'persuade_down', 'unreliable']


# ---------------------------------------------------------------------------
# JAX helpers
# ---------------------------------------------------------------------------

def _get_prior(s, prior):
    """Index into the prior array for state s."""
    return prior[s]


def _speaker_wpp(u, s, c, speaker_table):
    """P(u | state=(w,g), context=c).

    speaker_table has shape [N_STATES, n_ctx, n_utt].
    """
    return speaker_table[s, c, u]


# ---------------------------------------------------------------------------
# Memo model
# ---------------------------------------------------------------------------

@memo
def rsa_trust[s: WG, u: Utt](prior: ..., speaker_table: ..., c):
    """RSA listener: infer joint (world, goal) from observed utterance.

    Implements the vigilant listener from the persuasive RSA framework:
        P_L1(w, psi | u) proportional to P(w) * P(psi) * P_S1(u | w, psi, c)

    where psi in {informative, persuade_up, persuade_down, unreliable}.

    Returns |WG| x |Utt| array where [s, u] = P(state=s | observed u, context c).
    """
    listener: thinks[
        speaker: given(s in WG, wpp=_get_prior(s, prior)),
        speaker: chooses(u in Utt, wpp=_speaker_wpp(u, s, c, speaker_table)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(s in WG, wpp=Pr[speaker.s == s])
    return Pr[listener.s == s]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _compute_expected_outcome(dag: CausalDAG, effect_var: str,
                              context: Dict[str, int]) -> float:
    """Compute P(Y=1) under a DAG given context (as interventions).

    Used for persuasive speaker utility:
        V_pers+(u, c) = E[Y=1 | u's compressed DAG, c]
    """
    filtered = {k: v for k, v in context.items() if k in dag.variables}
    joint = dag.compute_joint(interventions=filtered)
    var_names = list(dag.variables.keys())
    e_idx = var_names.index(effect_var)
    return sum(prob for vals, prob in joint.items() if vals[e_idx] == 1)


# ---------------------------------------------------------------------------
# Default prior over speaker goals
# ---------------------------------------------------------------------------

DEFAULT_PRIOR_GOAL = {
    'informative': 1 / 4,
    'persuade_up': 1 / 4,
    'persuade_down': 1 / 4,
    'unreliable': 1 / 4,
}


# ---------------------------------------------------------------------------
# RSATrustModel wrapper
# ---------------------------------------------------------------------------

class RSATrustModel:
    """RSA-derived trust model with multiple speaker types.

    The listener jointly infers world complexity and speaker goal type:

        Speaker types:
        - Informative:    P(u|w,c) proportional to exp[-alpha * KL(P_true || P_compressed)]
        - Persuade-up:    P(u|w,c) proportional to exp[alpha * E[Y=1|u,c]]
        - Persuade-down:  P(u|w,c) proportional to exp[alpha * E[Y=0|u,c]]
        - Unreliable:     P(u|w,c) = 1/|U|  (uniform random)

        Vigilant listener:
        P_L1(w, psi | u) proportional to P(w) * P(psi) * P_S1(u | w, psi, c)

    The listener jointly infers:
    - world complexity (simple vs complex causal structure)
    - speaker type (informative, persuasive, or unreliable)
    """

    def __init__(
        self,
        world_dags: Dict[str, CausalDAG],
        utterances: List[Utterance],
        effect_var: str,
        prior_world: Dict[str, float],
        prior_goal: Dict[str, float] = None,
        speaker_alpha: float = 10.0,
        contexts: List[Dict[str, int]] = None,
    ):
        if len(utterances) != 2:
            raise ValueError(
                f"RSATrustModel requires exactly 2 utterances, got {len(utterances)}"
            )
        if len(world_dags) != 2:
            raise ValueError(
                f"RSATrustModel requires exactly 2 world hypotheses, got {len(world_dags)}"
            )

        self.utterances = utterances
        self.effect_var = effect_var
        self.prior_world = dict(prior_world)
        self.prior_goal = dict(prior_goal or DEFAULT_PRIOR_GOAL)
        self.speaker_alpha = speaker_alpha

        self._world_names = list(world_dags.keys())
        if contexts is None:
            contexts = [{}]
        self._contexts = contexts
        self._ctx_to_idx = {
            tuple(sorted(c.items())): i for i, c in enumerate(contexts)
        }

        # --- precompute speaker probability table [N_STATES, n_ctx, n_utt] ---
        n_worlds = len(self._world_names)
        n_ctx = len(contexts)
        n_utt = len(utterances)

        # Informative speaker: CompressionSpeaker (loss-based utility)
        self._speakers: Dict[str, CompressionSpeaker] = {}
        informative_table = np.zeros((n_worlds, n_ctx, n_utt))

        for wi, wname in enumerate(self._world_names):
            dag = world_dags[wname]
            spk = CompressionSpeaker(dag, utterances, effect_var, speaker_alpha)
            self._speakers[wname] = spk
            for ci, ctx in enumerate(contexts):
                filtered = {k: v for k, v in ctx.items() if k in dag.variables}
                probs = spk.get_utterance_probs(filtered)
                for ui, u in enumerate(utterances):
                    informative_table[wi, ci, ui] = probs[u.name]

        # Persuasive speakers: utility = expected outcome under compressed DAG
        persuade_up_table = np.zeros((n_worlds, n_ctx, n_utt))
        persuade_down_table = np.zeros((n_worlds, n_ctx, n_utt))

        for ci, ctx in enumerate(contexts):
            utils_up = np.zeros(n_utt)
            utils_down = np.zeros(n_utt)
            for ui, u in enumerate(utterances):
                p_y1 = _compute_expected_outcome(u.abstracted_dag, effect_var, ctx)
                utils_up[ui] = p_y1
                utils_down[ui] = 1.0 - p_y1

            probs_up = scipy_softmax(speaker_alpha * utils_up)
            probs_down = scipy_softmax(speaker_alpha * utils_down)

            # Same probabilities for all worlds (persuasive utility is
            # world-independent -- the speaker frames, not lies)
            for wi in range(n_worlds):
                persuade_up_table[wi, ci, :] = probs_up
                persuade_down_table[wi, ci, :] = probs_down

        # Unreliable speaker: uniform random, independent of world and context
        unreliable_table = np.full((n_worlds, n_ctx, n_utt), 1.0 / n_utt)

        # Pack into [N_STATES, n_ctx, n_utt]
        # State packing: s = w * N_GOALS + g
        goal_tables = {
            Goal.INFORMATIVE: informative_table,
            Goal.PERSUADE_UP: persuade_up_table,
            Goal.PERSUADE_DOWN: persuade_down_table,
            Goal.UNRELIABLE: unreliable_table,
        }
        speaker_table_np = np.zeros((N_STATES, n_ctx, n_utt))
        for wi in range(n_worlds):
            for gi in range(N_GOALS):
                si = wi * N_GOALS + gi
                speaker_table_np[si] = goal_tables[Goal(gi)][wi]

        self._speaker_table = jnp.array(speaker_table_np)

        # --- build initial prior over WG states ---
        prior_arr = np.zeros(N_STATES)
        for si in range(N_STATES):
            w = si // N_GOALS
            g = si % N_GOALS
            p_w = prior_world[self._world_names[w]]
            p_g = self.prior_goal[GOAL_NAMES[g]]
            prior_arr[si] = p_w * p_g

        self._initial_prior = jnp.array(prior_arr)
        self._current_prior = jnp.array(prior_arr)

    # -- helpers --------------------------------------------------------

    def _ctx_idx(self, context: Dict[str, int]) -> int:
        key = tuple(sorted(context.items()))
        return self._ctx_to_idx[key]

    def _utt_idx(self, utterance_name: str) -> int:
        for i, u in enumerate(self.utterances):
            if u.name == utterance_name:
                return i
        raise ValueError(f"Unknown utterance: {utterance_name}")

    # -- public API -----------------------------------------------------

    def update(
        self, observations: List[Tuple[dict, str]]
    ) -> Dict[str, float]:
        """Sequential Bayesian update via the memo RSA model.

        For each (context, utterance_name) observation:
          1. Call rsa_trust to get P(w,g | u, c) for all u.
          2. Extract the column for the observed u -> new prior.

        Returns dict with trust and complexity deltas.
        """
        prior_reliable = self.get_reliability_belief()
        prior_complex = self.get_complexity_belief()

        for ctx, utt_name in observations:
            c_idx = self._ctx_idx(ctx)
            u_idx = self._utt_idx(utt_name)

            # rsa_trust returns shape [|WG|, |Utt|]
            posterior_table = rsa_trust(
                prior=self._current_prior,
                speaker_table=self._speaker_table,
                c=c_idx,
            )
            # Column u_idx is the posterior P(s | observed u, c)
            self._current_prior = jnp.array(posterior_table[:, u_idx])

        posterior_reliable = self.get_reliability_belief()
        posterior_complex = self.get_complexity_belief()

        return {
            'trust_delta': float(posterior_reliable - prior_reliable),
            'complexity_delta': float(posterior_complex - prior_complex),
            'prior_reliable': float(prior_reliable),
            'posterior_reliable': float(posterior_reliable),
            'prior_complex': float(prior_complex),
            'posterior_complex': float(posterior_complex),
        }

    def update_with_explanation(
        self,
        observations: List[Tuple[dict, str]],
        explanation_strength: float = 1.0,
    ) -> Dict[str, float]:
        """Shift prior toward P(C=complex) before the Bayesian update.

        explanation_strength is in log-odds units: the complex-world
        hypotheses get exp(strength) more weight before normalising.
        """
        # mask: 1 for complex states (w=1), 0 for simple (w=0)
        complex_mask = jnp.array(
            [1.0 if si // N_GOALS == 1 else 0.0 for si in range(N_STATES)]
        )
        log_prior = jnp.log(self._current_prior + 1e-20)
        log_prior = log_prior + complex_mask * explanation_strength
        shifted = jnp.exp(log_prior)
        self._current_prior = shifted / shifted.sum()
        return self.update(observations)

    def get_reliability_belief(self) -> float:
        """P(speaker = informative) marginal."""
        p = np.asarray(self._current_prior)
        return float(sum(p[si] for si in range(N_STATES)
                        if si % N_GOALS == Goal.INFORMATIVE))

    def get_complexity_belief(self) -> float:
        """P(world = complex) marginal."""
        p = np.asarray(self._current_prior)
        return float(sum(p[si] for si in range(N_STATES)
                        if si // N_GOALS == 1))

    def get_goal_beliefs(self) -> Dict[str, float]:
        """Marginal beliefs over speaker goals."""
        p = np.asarray(self._current_prior)
        return {
            GOAL_NAMES[g]: float(sum(p[si] for si in range(N_STATES)
                                     if si % N_GOALS == g))
            for g in range(N_GOALS)
        }

    def get_beliefs(self) -> Dict[Tuple[str, str], float]:
        """Full normalized joint belief table."""
        p = np.asarray(self._current_prior)
        return {
            (self._world_names[si // N_GOALS],
             GOAL_NAMES[si % N_GOALS]): float(p[si])
            for si in range(N_STATES)
        }

    def reset(self):
        """Reset beliefs to prior."""
        self._current_prior = jnp.array(self._initial_prior)

    def get_derived_likelihoods(
        self,
        context: Dict[str, int],
        utterance_name: str,
    ) -> Dict[Tuple[str, str], float]:
        """Expose the computed likelihoods P(u | w, g, c)."""
        c_idx = self._ctx_idx(context)
        u_idx = self._utt_idx(utterance_name)
        result = {}
        for si in range(N_STATES):
            w = si // N_GOALS
            g = si % N_GOALS
            wname = self._world_names[w]
            glabel = GOAL_NAMES[g]
            result[(wname, glabel)] = float(self._speaker_table[si, c_idx, u_idx])
        return result


# ---------------------------------------------------------------------------
# Analysis tools
# ---------------------------------------------------------------------------

def compute_rate_distortion_curve(
    true_dag: CausalDAG,
    utterances: List[Utterance],
    effect_var: str,
    contexts: List[Dict[str, int]],
    alpha_range: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute the Rate-Distortion trade-off curve.

    The alpha parameter controls the trade-off:
    - alpha -> 0: Uniform distribution (minimal complexity, maximal distortion)
    - alpha -> inf: Deterministic optimal (maximal complexity, minimal distortion)
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

        avg_entropy = 0.0
        avg_distortion = 0.0

        for context in contexts:
            probs = speaker.get_utterance_probs(context)
            losses = speaker.compute_losses(context)

            p_array = np.array(list(probs.values()))
            p_array = p_array[p_array > 1e-10]
            entropy = -np.sum(p_array * np.log2(p_array))

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
    prior_goal: Optional[Dict[str, float]] = None,
    speaker_alpha: float = 10.0,
    contexts: List[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Sweep P(C=complex) from 0 to 1 and compute trust_delta at each value.

    Args:
        world_dags: {'simple': dag, 'complex': dag}
        utterances: Shared 2-utterance set
        effect_var: Outcome variable name
        observations: List of (context_dict, utterance_name) pairs
        prior_complex_range: Array of P(C=complex) values to sweep
        prior_goal: Prior over speaker goals (default: uniform over 4 types)
        speaker_alpha: Rationality parameter
        contexts: List of context dicts for speaker table precomputation
    """
    trust_deltas = []
    complexity_deltas = []

    for p_complex in prior_complex_range:
        model = RSATrustModel(
            world_dags=world_dags,
            utterances=utterances,
            effect_var=effect_var,
            prior_world={'simple': 1.0 - p_complex, 'complex': float(p_complex)},
            prior_goal=prior_goal,
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
