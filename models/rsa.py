import numpy as np
import jax.numpy as jnp
from memo import memo
from enum import IntEnum
from typing import Dict, List, Tuple

from scipy.special import softmax as scipy_softmax

from .dag import CausalDAG
from .speaker import Utterance, CompressionSpeaker


# ---------------------------------------------------------------------------
# Domains  (IntEnum, following the memo pattern)
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
# RSATrustModel  wrapper
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
        #   V_pers+(u, c) = P(Y=1 | u's DAG, c)    (inflate outcome belief)
        #   V_pers-(u, c) = P(Y=0 | u's DAG, c)    (deflate outcome belief)
        # These don't depend on the true world — only on what the
        # compressed DAG implies — matching the paper's formulation.
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
            # world-independent — the speaker frames, not lies)
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
