"""
Tests for causal compression model.
"""

import numpy as np
import pytest
from models import (
    Variable, CausalDAG, Utterance,
    compute_cmi, compute_cmi_multivar,
    compute_contextual_kl,
    CompressionSpeaker,
    RSATrustModel,
)


# ---------------------------------------------------------------------------
# Test fixture DAG builders (moved from examples.py)
# ---------------------------------------------------------------------------

def build_simple_medical_dag():
    """T -> Y, treatment has direct positive effect."""
    variables = {
        'T': Variable('T', (0, 1)),
        'Y': Variable('Y', (0, 1)),
    }
    parents = {'T': [], 'Y': ['T']}

    def cpt_T(p): return np.array([0.5, 0.5])
    def cpt_Y(p):
        return np.array([0.2, 0.8]) if p['T'] == 1 else np.array([0.7, 0.3])

    return CausalDAG(variables, parents, {'T': cpt_T, 'Y': cpt_Y})


def build_complex_medical_dag():
    """T -> Y <- M, treatment effect moderated by M."""
    variables = {
        'M': Variable('M', (0, 1)),
        'T': Variable('T', (0, 1)),
        'Y': Variable('Y', (0, 1)),
    }
    parents = {'M': [], 'T': [], 'Y': ['T', 'M']}

    def cpt_M(p): return np.array([0.5, 0.5])
    def cpt_T(p): return np.array([0.5, 0.5])
    def cpt_Y(p):
        t, m = p['T'], p['M']
        if m == 0:
            return np.array([0.1, 0.9]) if t == 1 else np.array([0.3, 0.7])
        else:
            return np.array([0.4, 0.6]) if t == 1 else np.array([0.8, 0.2])

    return CausalDAG(variables, parents, {'M': cpt_M, 'T': cpt_T, 'Y': cpt_Y})


def build_mask_advice_dag():
    """R, M -> I, mask effect varies by transmission rate."""
    variables = {
        'R': Variable('R', (0, 1)),
        'M': Variable('M', (0, 1)),
        'I': Variable('I', (0, 1)),
    }
    parents = {'R': [], 'M': [], 'I': ['R', 'M']}

    def cpt_R(p): return np.array([0.5, 0.5])
    def cpt_M(p): return np.array([0.5, 0.5])
    def cpt_I(p):
        r, m = p['R'], p['M']
        if r == 0:
            return np.array([0.95, 0.05]) if m == 1 else np.array([0.90, 0.10])
        else:
            return np.array([0.70, 0.30]) if m == 1 else np.array([0.30, 0.70])

    return CausalDAG(variables, parents, {'R': cpt_R, 'M': cpt_M, 'I': cpt_I})


def build_trust_update_scenario():
    """Returns dict with simple_dag, complex_dag, utterances, effect_var, contexts."""
    vars_dy = {'D': Variable('D', (0, 1)), 'Y': Variable('Y', (0, 1))}
    parents_dy = {'D': [], 'Y': ['D']}

    def cpt_D(p): return np.array([0.5, 0.5])

    def cpt_Y_works(p):
        return np.array([0.1, 0.9]) if p['D'] == 1 else np.array([0.5, 0.5])

    def cpt_Y_doesnt(p):
        return np.array([0.8, 0.2]) if p['D'] == 1 else np.array([0.5, 0.5])

    dag_works = CausalDAG(vars_dy, parents_dy, {'D': cpt_D, 'Y': cpt_Y_works})
    dag_doesnt = CausalDAG(vars_dy, parents_dy, {'D': cpt_D, 'Y': cpt_Y_doesnt})

    utterances = [
        Utterance("drug_works", dag_works, "stability"),
        Utterance("drug_doesnt_work", dag_doesnt, "stability"),
    ]

    def cpt_Y_simple(p):
        return np.array([0.15, 0.85]) if p['D'] == 1 else np.array([0.55, 0.45])
    simple_dag = CausalDAG(vars_dy, parents_dy, {'D': cpt_D, 'Y': cpt_Y_simple})

    vars_gdy = {
        'G': Variable('G', (0, 1)),
        'D': Variable('D', (0, 1)),
        'Y': Variable('Y', (0, 1)),
    }
    parents_gdy = {'G': [], 'D': [], 'Y': ['G', 'D']}

    def cpt_G(p): return np.array([0.5, 0.5])
    def cpt_Y_complex(p):
        g, d = p['G'], p['D']
        if   g == 1 and d == 1: return np.array([0.1, 0.9])
        elif g == 1 and d == 0: return np.array([0.4, 0.6])
        elif g == 0 and d == 1: return np.array([0.8, 0.2])
        else:                   return np.array([0.6, 0.4])
    complex_dag = CausalDAG(vars_gdy, parents_gdy,
                            {'G': cpt_G, 'D': cpt_D, 'Y': cpt_Y_complex})

    return {
        'simple_dag': simple_dag,
        'complex_dag': complex_dag,
        'utterances': utterances,
        'effect_var': 'Y',
        'contexts': [{'G': 0}, {'G': 1}],
    }


class TestCausalDAG:
    """Test the CausalDAG class."""

    def test_simple_dag_creation(self):
        """Test creating a simple 2-variable DAG."""
        dag = build_simple_medical_dag()
        assert len(dag.variables) == 2
        assert 'T' in dag.variables
        assert 'Y' in dag.variables

    def test_complex_dag_creation(self):
        """Test creating a 3-variable DAG with moderator."""
        dag = build_complex_medical_dag()
        assert len(dag.variables) == 3
        assert 'M' in dag.variables
        assert dag.parents['Y'] == ['T', 'M']

    def test_topological_order(self):
        """Test that topological order respects parent relationships."""
        dag = build_complex_medical_dag()
        order = dag.get_topological_order()
        # Parents must come before children
        y_idx = order.index('Y')
        t_idx = order.index('T')
        m_idx = order.index('M')
        assert t_idx < y_idx
        assert m_idx < y_idx

    def test_sampling(self):
        """Test sampling from DAG."""
        dag = build_simple_medical_dag()
        samples = dag.sample(n=100)
        assert len(samples) == 100
        assert all('T' in s and 'Y' in s for s in samples)
        assert all(s['T'] in (0, 1) and s['Y'] in (0, 1) for s in samples)

    def test_intervention_sampling(self):
        """Test sampling with do() intervention."""
        dag = build_simple_medical_dag()
        samples = dag.sample(n=100, interventions={'T': 1})
        assert all(s['T'] == 1 for s in samples)

    def test_joint_distribution_sums_to_one(self):
        """Test that joint distribution is normalized."""
        dag = build_complex_medical_dag()
        joint = dag.compute_joint()
        total = sum(joint.values())
        assert np.isclose(total, 1.0, atol=1e-10)

    def test_interventional_joint(self):
        """Test joint distribution with intervention."""
        dag = build_complex_medical_dag()
        joint_do_t1 = dag.compute_joint(interventions={'T': 1})

        # All entries with T=0 should have probability 0
        var_names = list(dag.variables.keys())
        t_idx = var_names.index('T')

        for vals, prob in joint_do_t1.items():
            if vals[t_idx] == 0:
                assert prob == 0.0
            # Check normalization among T=1 entries
        assert np.isclose(sum(joint_do_t1.values()), 1.0, atol=1e-10)


class TestCMI:
    """Test Causal Mutual Information computation."""

    def test_cmi_positive(self):
        """CMI should be non-negative."""
        dag = build_simple_medical_dag()
        cmi = compute_cmi(dag, 'T', 'Y')
        assert cmi >= 0

    def test_cmi_independent_is_zero(self):
        """CMI should be 0 for independent variables."""
        # Create DAG where T and Y are independent
        variables = {
            'T': Variable('T', (0, 1)),
            'Y': Variable('Y', (0, 1)),
        }
        parents = {'T': [], 'Y': []}  # No causal connection

        def cpt_T(p): return np.array([0.5, 0.5])
        def cpt_Y(p): return np.array([0.5, 0.5])

        dag = CausalDAG(variables, parents, {'T': cpt_T, 'Y': cpt_Y})
        cmi = compute_cmi(dag, 'T', 'Y')
        assert np.isclose(cmi, 0.0, atol=1e-10)

    def test_cmi_deterministic_is_one(self):
        """CMI should be 1 bit for deterministic binary relationship."""
        variables = {
            'T': Variable('T', (0, 1)),
            'Y': Variable('Y', (0, 1)),
        }
        parents = {'T': [], 'Y': ['T']}

        def cpt_T(p): return np.array([0.5, 0.5])
        def cpt_Y(p):
            # Y = T (deterministic)
            t = p['T']
            if t == 0:
                return np.array([1.0, 0.0])
            else:
                return np.array([0.0, 1.0])

        dag = CausalDAG(variables, parents, {'T': cpt_T, 'Y': cpt_Y})
        cmi = compute_cmi(dag, 'T', 'Y')
        assert np.isclose(cmi, 1.0, atol=1e-6)


class TestContextualKL:
    """Test context-conditioned KL divergence loss."""

    def test_loss_varies_by_context(self):
        """KL loss should differ for different contexts."""
        dag = build_mask_advice_dag()

        # Simple abstraction (ignores R)
        vars_simple = {
            'M': Variable('M', (0, 1)),
            'I': Variable('I', (0, 1)),
        }
        parents_simple = {'M': [], 'I': ['M']}

        def cpt_M(p): return np.array([0.5, 0.5])
        def cpt_I(p):
            m = p['M']
            # Average effect
            if m == 1:
                return np.array([0.825, 0.175])
            else:
                return np.array([0.6, 0.4])

        simple_dag = CausalDAG(vars_simple, parents_simple, {'M': cpt_M, 'I': cpt_I})

        # Compute loss in different contexts
        loss_low_R = compute_contextual_kl(dag, simple_dag, 'I', {'R': 0})
        loss_high_R = compute_contextual_kl(dag, simple_dag, 'I', {'R': 1})

        # Losses should be different
        assert loss_low_R != loss_high_R

    def test_zero_loss_for_identical_model(self):
        """KL should be 0 when true and compressed models agree."""
        dag = build_simple_medical_dag()
        kl = compute_contextual_kl(dag, dag, 'Y', {'T': 1})
        assert np.isclose(kl, 0.0, atol=1e-10)


class TestCompressionSpeaker:
    """Test the CompressionSpeaker class."""

    def test_utterance_probs_sum_to_one(self):
        """Utterance probabilities should sum to 1."""
        dag = build_mask_advice_dag()

        vars_simple = {
            'M': Variable('M', (0, 1)),
            'I': Variable('I', (0, 1)),
        }
        parents_simple = {'M': [], 'I': ['M']}

        def cpt_M(p): return np.array([0.5, 0.5])
        def cpt_I(p): return np.array([0.7, 0.3]) if p['M'] == 1 else np.array([0.5, 0.5])

        simple_dag = CausalDAG(vars_simple, parents_simple, {'M': cpt_M, 'I': cpt_I})

        utterances = [
            Utterance("full", dag, "full"),
            Utterance("simple", simple_dag, "stability"),
        ]

        speaker = CompressionSpeaker(dag, utterances, 'I', alpha=1.0)
        probs = speaker.get_utterance_probs({'R': 0})

        assert np.isclose(sum(probs.values()), 1.0, atol=1e-10)

    def test_higher_alpha_more_deterministic(self):
        """Higher alpha should make choice more deterministic."""
        dag = build_mask_advice_dag()

        vars_simple = {
            'M': Variable('M', (0, 1)),
            'I': Variable('I', (0, 1)),
        }
        parents_simple = {'M': [], 'I': ['M']}

        def cpt_M(p): return np.array([0.5, 0.5])
        def cpt_I_good(p): return np.array([0.9, 0.1]) if p['M'] == 1 else np.array([0.9, 0.1])
        def cpt_I_bad(p): return np.array([0.2, 0.8]) if p['M'] == 1 else np.array([0.2, 0.8])

        good_dag = CausalDAG(vars_simple, parents_simple, {'M': cpt_M, 'I': cpt_I_good})
        bad_dag = CausalDAG(vars_simple, parents_simple, {'M': cpt_M, 'I': cpt_I_bad})

        utterances = [
            Utterance("full", dag, "full"),
            Utterance("good", good_dag, "stability"),
            Utterance("bad", bad_dag, "stability"),
        ]

        speaker_low_alpha = CompressionSpeaker(dag, utterances, 'I', alpha=0.1)
        speaker_high_alpha = CompressionSpeaker(dag, utterances, 'I', alpha=10.0)

        probs_low = speaker_low_alpha.get_utterance_probs({'R': 0})
        probs_high = speaker_high_alpha.get_utterance_probs({'R': 0})

        # Higher alpha should have more extreme probabilities
        max_prob_low = max(probs_low.values())
        max_prob_high = max(probs_high.values())

        assert max_prob_high >= max_prob_low


class TestContextDependentCompression:
    """Test the key prediction: same DAG produces different utterances in different contexts."""

    def test_context_changes_optimal_utterance(self):
        """
        Main test: verify that context affects which compressed utterance is optimal.
        """
        # Build true DAG with context-dependent effects
        variables = {
            'G': Variable('G', (0, 1)),  # Context (genetic marker)
            'D': Variable('D', (0, 1)),  # Treatment
            'Y': Variable('Y', (0, 1)),  # Outcome
        }
        parents = {'G': [], 'D': [], 'Y': ['G', 'D']}

        def cpt_G(p): return np.array([0.5, 0.5])
        def cpt_D(p): return np.array([0.5, 0.5])
        def cpt_Y(p):
            g, d = p['G'], p['D']
            if g == 1 and d == 1:
                return np.array([0.1, 0.9])  # Treatment works when G=1
            elif g == 0 and d == 1:
                return np.array([0.9, 0.1])  # Treatment fails when G=0
            else:
                return np.array([0.5, 0.5])

        true_dag = CausalDAG(variables, parents, {'G': cpt_G, 'D': cpt_D, 'Y': cpt_Y})

        # Compressed utterances
        vars_simple = {'D': Variable('D', (0, 1)), 'Y': Variable('Y', (0, 1))}
        parents_simple = {'D': [], 'Y': ['D']}

        # "Treatment works"
        def cpt_Y_works(p):
            return np.array([0.1, 0.9]) if p['D'] == 1 else np.array([0.5, 0.5])
        dag_works = CausalDAG(vars_simple, parents_simple, {'D': cpt_D, 'Y': cpt_Y_works})

        # "Treatment doesn't work"
        def cpt_Y_fails(p):
            return np.array([0.9, 0.1]) if p['D'] == 1 else np.array([0.5, 0.5])
        dag_fails = CausalDAG(vars_simple, parents_simple, {'D': cpt_D, 'Y': cpt_Y_fails})

        utterances = [
            Utterance("works", dag_works, "stability"),
            Utterance("fails", dag_fails, "stability"),
        ]

        speaker = CompressionSpeaker(true_dag, utterances, 'Y', alpha=10.0)

        # Context 1: G=1 (treatment should work)
        losses_g1 = speaker.compute_losses({'G': 1})
        # Context 2: G=0 (treatment should fail)
        losses_g0 = speaker.compute_losses({'G': 0})

        # In G=1, "works" should have lower loss than "fails"
        assert losses_g1['works'] < losses_g1['fails']

        # In G=0, "fails" should have lower loss than "works"
        assert losses_g0['fails'] < losses_g0['works']

        # Optimal utterance should change with context
        optimal_g1 = speaker.get_optimal_utterance({'G': 1})
        optimal_g0 = speaker.get_optimal_utterance({'G': 0})

        assert optimal_g1.name == 'works'
        assert optimal_g0.name == 'fails'


class TestRSATrustModel:
    """Test the RSA trust model with persuasive speaker types."""

    @pytest.fixture
    def scenario(self):
        return build_trust_update_scenario()

    @pytest.fixture
    def model(self, scenario):
        return RSATrustModel(
            world_dags={
                'simple': scenario['simple_dag'],
                'complex': scenario['complex_dag'],
            },
            utterances=scenario['utterances'],
            effect_var=scenario['effect_var'],
            prior_world={'simple': 0.5, 'complex': 0.5},
            speaker_alpha=10.0,
            contexts=scenario['contexts'],
        )

    def _make_model(self, scenario, prior_world, **kwargs):
        """Helper to create RSATrustModel with given priors."""
        defaults = dict(
            world_dags={
                'simple': scenario['simple_dag'],
                'complex': scenario['complex_dag'],
            },
            utterances=scenario['utterances'],
            effect_var='Y',
            prior_world=prior_world,
            speaker_alpha=10.0,
            contexts=scenario['contexts'],
        )
        defaults.update(kwargs)
        return RSATrustModel(**defaults)

    def test_beliefs_normalize(self, model):
        """Joint beliefs over (world, goal) should sum to 1."""
        beliefs = model.get_beliefs()
        total = sum(beliefs.values())
        assert np.isclose(total, 1.0, atol=1e-6)
        assert len(beliefs) == 8  # 2 worlds x 4 goals

    def test_initial_marginals(self, model):
        """Initial marginals should match the priors."""
        # Default prior_goal is uniform 1/4 each, so P(informative) = 1/4
        assert np.isclose(model.get_reliability_belief(), 1 / 4, atol=1e-6)
        assert np.isclose(model.get_complexity_belief(), 0.5, atol=1e-6)

    def test_goal_beliefs(self, model):
        """Goal beliefs should match prior and sum to 1."""
        goals = model.get_goal_beliefs()
        assert set(goals.keys()) == {'informative', 'persuade_up', 'persuade_down', 'unreliable'}
        assert np.isclose(sum(goals.values()), 1.0, atol=1e-6)
        for g in goals.values():
            assert np.isclose(g, 1 / 4, atol=1e-6)

    def test_unreliable_speaker_uniform(self, model):
        """Unreliable speaker likelihoods should be uniform (0.5 each)."""
        for ctx in [{'G': 0}, {'G': 1}]:
            lk = model.get_derived_likelihoods(ctx, 'drug_works')
            assert np.isclose(lk[('simple', 'unreliable')], 0.5, atol=1e-6)
            assert np.isclose(lk[('complex', 'unreliable')], 0.5, atol=1e-6)

    def test_informative_likelihoods_match_speaker(self, scenario, model):
        """Informative speaker likelihoods should match CompressionSpeaker probs."""
        ctx = {'G': 1}
        lk = model.get_derived_likelihoods(ctx, 'drug_works')

        spk = CompressionSpeaker(
            scenario['complex_dag'],
            scenario['utterances'],
            scenario['effect_var'],
            alpha=10.0,
        )
        probs = spk.get_utterance_probs(ctx)
        assert np.isclose(
            lk[('complex', 'informative')],
            probs['drug_works'],
            atol=1e-6,
        )

    def test_persuasive_speaker_directional(self, model):
        """Persuade-up should prefer drug_works; persuade-down should prefer drug_doesnt_work."""
        lk_works = model.get_derived_likelihoods({'G': 1}, 'drug_works')
        lk_doesnt = model.get_derived_likelihoods({'G': 1}, 'drug_doesnt_work')

        # Persuade-up strongly prefers drug_works
        assert lk_works[('simple', 'persuade_up')] > 0.9
        assert lk_doesnt[('simple', 'persuade_up')] < 0.1

        # Persuade-down strongly prefers drug_doesnt_work
        assert lk_doesnt[('simple', 'persuade_down')] > 0.9
        assert lk_works[('simple', 'persuade_down')] < 0.1

        # Persuasive probs are world-independent
        assert np.isclose(
            lk_works[('simple', 'persuade_up')],
            lk_works[('complex', 'persuade_up')],
            atol=1e-6,
        )

    def test_revision_signals_complex_informative(self, scenario):
        """Revision (changing advice across contexts) should increase
        P(complex) and P(informative) when priors are moderate."""
        model = self._make_model(scenario, {'simple': 0.5, 'complex': 0.5})
        result = model.update([
            ({'G': 1}, 'drug_works'),
            ({'G': 0}, 'drug_doesnt_work'),
        ])
        # Revision uniquely fingerprints (complex, informative)
        assert result['trust_delta'] > 0
        assert result['complexity_delta'] > 0

    def test_consistency_ambiguous(self, scenario):
        """Consistent advice is ambiguous: could be (simple, informative)
        or (any world, persuade_up).  For a complex-believer, trust
        should decrease because informative+complex would have revised."""
        complex_model = self._make_model(scenario, {'simple': 0.1, 'complex': 0.9})
        result = complex_model.update([
            ({'G': 1}, 'drug_works'),
            ({'G': 0}, 'drug_works'),
        ])
        # Complex-believer hearing consistent advice: trust decreases
        # (an informative speaker in a complex world would have revised)
        assert result['trust_delta'] < 0

    def test_consistency_simple_believer_trusts(self, scenario):
        """For a simple-believer, consistent advice supports (simple, informative)
        and trust should increase."""
        simple_model = self._make_model(scenario, {'simple': 0.9, 'complex': 0.1})
        result = simple_model.update([
            ({'G': 1}, 'drug_works'),
            ({'G': 0}, 'drug_works'),
        ])
        assert result['trust_delta'] > 0

    def test_crossover_exists_consistent(self, scenario):
        """Sweeping P(complex) with consistent observations should produce
        a sign change in trust_delta."""
        observations = [
            ({'G': 1}, 'drug_works'),
            ({'G': 0}, 'drug_works'),
        ]
        prior_range = np.linspace(0.05, 0.95, 30)
        deltas = []
        for p_complex in prior_range:
            m = self._make_model(scenario, {
                'simple': 1.0 - p_complex, 'complex': float(p_complex),
            })
            deltas.append(m.update(observations)['trust_delta'])

        deltas = np.array(deltas)
        sign_changes = np.where(np.diff(np.sign(deltas)))[0]
        assert len(sign_changes) > 0, "Expected a crossover in trust_delta"

    def test_explanation_preserves_trust(self, scenario):
        """Explanation (shifting prior toward complex) should increase
        trust delta for revision observations."""
        observations = [
            ({'G': 1}, 'drug_works'),
            ({'G': 0}, 'drug_doesnt_work'),
        ]

        model_no = self._make_model(scenario, {'simple': 0.5, 'complex': 0.5})
        model_exp = self._make_model(scenario, {'simple': 0.5, 'complex': 0.5})

        r_no = model_no.update(observations)
        r_exp = model_exp.update_with_explanation(observations, explanation_strength=2.0)

        # With explanation, trust_delta should be at least as high
        assert r_exp['trust_delta'] >= r_no['trust_delta'] - 0.01

    def test_result_format(self, model):
        """update() should return the expected dict keys."""
        result = model.update([({'G': 1}, 'drug_works')])
        expected_keys = {
            'trust_delta', 'complexity_delta',
            'prior_reliable', 'posterior_reliable',
            'prior_complex', 'posterior_complex',
        }
        assert set(result.keys()) == expected_keys

    def test_reset(self, model):
        """reset() should restore initial beliefs."""
        model.update([({'G': 1}, 'drug_works')])
        model.reset()
        assert np.isclose(model.get_reliability_belief(), 1 / 4, atol=1e-6)
        assert np.isclose(model.get_complexity_belief(), 0.5, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
