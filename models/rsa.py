import numpy as np
import jax.numpy as jnp
from memo import memo
from enum import IntEnum
from typing import Dict, List, Tuple

from .dag import CausalDAG
from .speaker import Utterance, CompressionSpeaker


# ---------------------------------------------------------------------------
# Domains  (IntEnum, following Hannah's rsa_memo.py pattern)
# ---------------------------------------------------------------------------

class Utt(IntEnum):
    """Two-utterance domain for trust scenarios."""
    U0 = 0
    U1 = 1


class WR(IntEnum):
    """Joint state: (world_type, reliability) packed into a single index.
       w: 0 = simple, 1 = complex
       r: 0 = reliable, 1 = unreliable
       Packing: s = w * 2 + r  ->  w = s // 2,  r = s % 2
    """
    SIMPLE_RELIABLE = 0
    SIMPLE_UNRELIABLE = 1
    COMPLEX_RELIABLE = 2
    COMPLEX_UNRELIABLE = 3


# ---------------------------------------------------------------------------
# JAX helpers
# ---------------------------------------------------------------------------

def _get_prior(s, prior):
    """Index into the prior array for state s."""
    return prior[s]


def _speaker_wpp(u, s, c, speaker_table):
    """P(u | state=(w,r), context=c).

    Reliable speaker (r=0): probability from precomputed CompressionSpeaker table.
    Unreliable speaker (r=1): uniform 1/|U| = 0.5.
    """
    w = s // 2
    r = s % 2
    return jnp.where(r == 0, speaker_table[w, c, u], 0.5)


# ---------------------------------------------------------------------------
# Memo model
# ---------------------------------------------------------------------------

@memo
def rsa_trust[s: WR, u: Utt](prior: ..., speaker_table: ..., c):
    """RSA listener: infer joint (world, reliability) from observed utterance.

    Returns |WR| x |Utt| array where [s, u] = P(state=s | observed u, context c).
    """
    listener: thinks[
        speaker: given(s in WR, wpp=_get_prior(s, prior)),
        speaker: chooses(u in Utt, wpp=_speaker_wpp(u, s, c, speaker_table)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(s in WR, wpp=Pr[speaker.s == s])
    return Pr[listener.s == s]


# ---------------------------------------------------------------------------
# RSATrustModel  wrapper
# ---------------------------------------------------------------------------

N_STATES = len(WR)


class RSATrustModel:
    """RSA-derived trust model.

    Likelihoods are computed from CompressionSpeaker (not hand-coded).
    Inference is performed via the memo DSL, which compiles the RSA model
    to JAX and handles Bayesian normalization automatically.

    Sequential observations are handled by feeding posteriors back as priors.
    """

    def __init__(
        self,
        world_dags: Dict[str, CausalDAG],
        utterances: List[Utterance],
        effect_var: str,
        prior_world: Dict[str, float],
        prior_reliable: float = 0.8,
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
        self.prior_reliable = prior_reliable
        self.speaker_alpha = speaker_alpha

        self._world_names = list(world_dags.keys())
        if contexts is None:
            contexts = [{}]
        self._contexts = contexts
        self._ctx_to_idx = {
            tuple(sorted(c.items())): i for i, c in enumerate(contexts)
        }

        # --- precompute speaker probability table [n_worlds, n_ctx, n_utt] ---
        n_worlds = len(self._world_names)
        n_ctx = len(contexts)
        n_utt = len(utterances)

        self._speakers: Dict[str, CompressionSpeaker] = {}
        speaker_table_np = np.zeros((n_worlds, n_ctx, n_utt))

        for wi, wname in enumerate(self._world_names):
            dag = world_dags[wname]
            spk = CompressionSpeaker(dag, utterances, effect_var, speaker_alpha)
            self._speakers[wname] = spk
            for ci, ctx in enumerate(contexts):
                filtered = {k: v for k, v in ctx.items() if k in dag.variables}
                probs = spk.get_utterance_probs(filtered)
                for ui, u in enumerate(utterances):
                    speaker_table_np[wi, ci, ui] = probs[u.name]

        self._speaker_table = jnp.array(speaker_table_np)

        # --- build initial prior over WR states ---
        prior_arr = np.zeros(N_STATES)
        for si in range(N_STATES):
            w = si // 2
            r = si % 2
            p_w = prior_world[self._world_names[w]]
            p_r = prior_reliable if r == 0 else (1 - prior_reliable)
            prior_arr[si] = p_w * p_r

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
          1. Call rsa_trust to get P(w,r | u, c) for all u.
          2. Extract the column for the observed u -> new prior.

        Returns dict matching TrustModel.update() format.
        """
        prior_reliable = self.get_reliability_belief()
        prior_complex = self.get_complexity_belief()

        for ctx, utt_name in observations:
            c_idx = self._ctx_idx(ctx)
            u_idx = self._utt_idx(utt_name)

            # rsa_trust returns shape [|WR|, |Utt|]
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
        # mask: 1 for complex states, 0 for simple
        complex_mask = jnp.array(
            [1.0 if si // 2 == 1 else 0.0 for si in range(N_STATES)]
        )
        log_prior = jnp.log(self._current_prior + 1e-20)
        log_prior = log_prior + complex_mask * explanation_strength
        shifted = jnp.exp(log_prior)
        self._current_prior = shifted / shifted.sum()
        return self.update(observations)

    def get_reliability_belief(self) -> float:
        """P(speaker = reliable) marginal."""
        p = np.asarray(self._current_prior)
        return float(sum(p[si] for si in range(N_STATES) if si % 2 == 0))

    def get_complexity_belief(self) -> float:
        """P(world = complex) marginal."""
        p = np.asarray(self._current_prior)
        return float(sum(p[si] for si in range(N_STATES) if si // 2 == 1))

    def get_beliefs(self) -> Dict[Tuple[str, str], float]:
        """Full normalized joint belief table."""
        p = np.asarray(self._current_prior)
        return {
            (self._world_names[si // 2],
             'reliable' if si % 2 == 0 else 'unreliable'): float(p[si])
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
        """Expose the computed likelihoods P(u | C, R, c) for comparison with
        the hand-coded values in TrustModel.
        """
        c_idx = self._ctx_idx(context)
        u_idx = self._utt_idx(utterance_name)
        result = {}
        for si in range(N_STATES):
            w = si // 2
            r = si % 2
            wname = self._world_names[w]
            rlabel = 'reliable' if r == 0 else 'unreliable'
            if r == 0:
                result[(wname, rlabel)] = float(self._speaker_table[w, c_idx, u_idx])
            else:
                result[(wname, rlabel)] = 1.0 / len(self.utterances)
        return result
