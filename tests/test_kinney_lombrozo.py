"""
Verification tests for Kinney & Lombrozo (2024) information-theoretic framework.

Tests reproduce the exact numerical examples from the paper:
- Footnote 10 (p.9): CMI and information loss for Bricofly experiment
- Table 2 (p.15): Information loss values for Experiment 1
- Experiment 2: Information loss values with different base rate
- Compression formula: p(e|do(c')) = sum over sigma^{-1}(c') of p(e|do(c)) * q(c)/q(c')
"""

import numpy as np
import pytest
from models import (
    Variable, CausalDAG,
    compute_cmi, compute_cmi_multivar,
    compute_information_loss, compress_dag,
)


# ---------------------------------------------------------------------------
# Helper: build Bricofly DAGs from the paper
# ---------------------------------------------------------------------------

def build_bricofly_detailed(x: float):
    """
    Build the LESS compressed Bricofly DAG (4-value cause).

    Tank Condition: {WarmHumid, WarmDry, ColdHumid, ColdDry} = {0, 1, 2, 3}
    Blue Wings: {No, Yes} = {0, 1}

    Data (Experiment 1, Report 1, proportionality condition):
        P(BW=Yes | do(WarmHumid)) = x
        P(BW=Yes | do(WarmDry))   = 0.70
        P(BW=Yes | do(ColdHumid)) = 0.01
        P(BW=Yes | do(ColdDry))   = 0.01
    """
    variables = {
        'TC': Variable('TC', (0, 1, 2, 3)),
        'BW': Variable('BW', (0, 1)),
    }
    parents = {'TC': [], 'BW': ['TC']}

    def cpt_TC(p):
        return np.array([0.25, 0.25, 0.25, 0.25])

    def cpt_BW(p):
        tc = p['TC']
        probs_bw_yes = {0: x, 1: 0.70, 2: 0.01, 3: 0.01}
        p_yes = probs_bw_yes[tc]
        return np.array([1.0 - p_yes, p_yes])

    return CausalDAG(variables, parents, {'TC': cpt_TC, 'BW': cpt_BW})


# Compression map: {WH, WD} -> Warm (0), {CH, CD} -> Cold (1)
BRICOFLY_COMPRESSION = {0: 0, 1: 0, 2: 1, 3: 1}


def build_bricofly_compressed_manual(x: float):
    """
    Build the MORE compressed Bricofly DAG by hand (for comparison).

    Tank: {Warm, Cold} = {0, 1}
    Blue Wings: {No, Yes} = {0, 1}

    The averaged interventional distributions should be:
        P(BW=Yes | do(Warm)) = (x + 0.70) / 2
        P(BW=Yes | do(Cold)) = (0.01 + 0.01) / 2 = 0.01
    """
    avg_warm = (x + 0.70) / 2.0

    variables = {
        'T': Variable('T', (0, 1)),
        'BW': Variable('BW', (0, 1)),
    }
    parents = {'T': [], 'BW': ['T']}

    def cpt_T(p):
        return np.array([0.5, 0.5])

    def cpt_BW(p):
        t = p['T']
        if t == 0:  # Warm
            return np.array([1.0 - avg_warm, avg_warm])
        else:       # Cold
            return np.array([0.99, 0.01])

    return CausalDAG(variables, parents, {'T': cpt_T, 'BW': cpt_BW})


def compute_cmi_manual(probs_effect_given_cause):
    """
    Manually compute CMI from a table of interventional probabilities.

    Independent check against the DAG-based computation.

    Args:
        probs_effect_given_cause: dict mapping cause_value -> list of P(e|do(c))
            for each effect value. Uniform q assumed.
    """
    cause_values = list(probs_effect_given_cause.keys())
    n_causes = len(cause_values)
    q = 1.0 / n_causes
    n_effects = len(probs_effect_given_cause[cause_values[0]])

    p_e = np.zeros(n_effects)
    for c in cause_values:
        for ei in range(n_effects):
            p_e[ei] += q * probs_effect_given_cause[c][ei]

    cmi = 0.0
    for c in cause_values:
        for ei in range(n_effects):
            p_ec = probs_effect_given_cause[c][ei]
            if p_ec > 0 and p_e[ei] > 0:
                cmi += q * p_ec * np.log2(p_ec / p_e[ei])

    return cmi


# ---------------------------------------------------------------------------
# Tests: compress_dag produces correct CPTs
# ---------------------------------------------------------------------------

class TestCompressDag:
    """
    Verify that compress_dag mechanically applies the paper's formula:

        p(e|do(c')) = sum_{c in sigma^{-1}(c')} p(e|do(c)) * q(c) / q(c')

    and that the resulting DAG matches hand-built compressed DAGs.
    """

    @pytest.mark.parametrize("x", [0.70, 0.85, 0.98])
    def test_compressed_cpt_warm(self, x):
        """P(BW=1|do(Warm)) should equal (x + 0.70) / 2."""
        dag = build_bricofly_detailed(x)
        compressed = compress_dag(dag, 'TC', 'BW', BRICOFLY_COMPRESSION,
                                  compressed_cause_name='T')

        joint_warm = compressed.compute_joint(interventions={'T': 0})
        var_names = list(compressed.variables.keys())
        bw_idx = var_names.index('BW')

        p_bw1_warm = sum(p for v, p in joint_warm.items() if v[bw_idx] == 1)
        expected = (x + 0.70) / 2.0
        assert np.isclose(p_bw1_warm, expected, atol=1e-10), \
            f"P(BW=1|do(Warm)) = {p_bw1_warm}, expected {expected}"

    @pytest.mark.parametrize("x", [0.70, 0.85, 0.98])
    def test_compressed_cpt_cold(self, x):
        """P(BW=1|do(Cold)) should equal (0.01 + 0.01) / 2 = 0.01."""
        dag = build_bricofly_detailed(x)
        compressed = compress_dag(dag, 'TC', 'BW', BRICOFLY_COMPRESSION,
                                  compressed_cause_name='T')

        joint_cold = compressed.compute_joint(interventions={'T': 1})
        var_names = list(compressed.variables.keys())
        bw_idx = var_names.index('BW')

        p_bw1_cold = sum(p for v, p in joint_cold.items() if v[bw_idx] == 1)
        assert np.isclose(p_bw1_cold, 0.01, atol=1e-10), \
            f"P(BW=1|do(Cold)) = {p_bw1_cold}, expected 0.01"

    @pytest.mark.parametrize("x", [0.70, 0.85, 0.98])
    def test_compressed_matches_manual(self, x):
        """compress_dag output should match the hand-built compressed DAG."""
        dag = build_bricofly_detailed(x)
        auto = compress_dag(dag, 'TC', 'BW', BRICOFLY_COMPRESSION,
                            compressed_cause_name='T')
        manual = build_bricofly_compressed_manual(x)

        # Compare CMI values (which depend on the full CPT)
        cmi_auto = compute_cmi(auto, 'T', 'BW')
        cmi_manual = compute_cmi(manual, 'T', 'BW')
        assert np.isclose(cmi_auto, cmi_manual, atol=1e-10), \
            f"Auto CMI ({cmi_auto}) != Manual CMI ({cmi_manual})"

    @pytest.mark.parametrize("x", [0.70, 0.85, 0.98])
    def test_compressed_joint_normalized(self, x):
        """Compressed DAG joint should sum to 1."""
        dag = build_bricofly_detailed(x)
        compressed = compress_dag(dag, 'TC', 'BW', BRICOFLY_COMPRESSION,
                                  compressed_cause_name='T')
        joint = compressed.compute_joint()
        assert np.isclose(sum(joint.values()), 1.0, atol=1e-10)

    def test_identity_compression(self):
        """Compression that doesn't merge anything should preserve CMI."""
        dag = build_bricofly_detailed(0.85)
        # Identity map: each value maps to itself
        identity = {0: 0, 1: 1, 2: 2, 3: 3}
        compressed = compress_dag(dag, 'TC', 'BW', identity,
                                  compressed_cause_name='TC2')

        cmi_orig = compute_cmi(dag, 'TC', 'BW')
        cmi_comp = compute_cmi(compressed, 'TC2', 'BW')
        assert np.isclose(cmi_orig, cmi_comp, atol=1e-10)

    def test_full_compression_zero_cmi(self):
        """Compressing all values to one should give CMI = 0."""
        dag = build_bricofly_detailed(0.85)
        # All values map to 0
        full_compress = {0: 0, 1: 0, 2: 0, 3: 0}
        compressed = compress_dag(dag, 'TC', 'BW', full_compress,
                                  compressed_cause_name='T')

        cmi = compute_cmi(compressed, 'T', 'BW')
        assert np.isclose(cmi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: compute_information_loss matches paper values
# ---------------------------------------------------------------------------

class TestInformationLoss:
    """
    Verify compute_information_loss against the paper's reported values.

    Uses compress_dag to build the compressed DAG, then checks that
    L = CMI(detailed) - CMI(compressed) matches the paper.
    """

    @pytest.mark.parametrize("x,expected_loss", [
        (0.70, 0.00),
        (0.85, 0.01),
        (0.98, 0.06),
    ])
    def test_experiment1_losses(self, x, expected_loss):
        """Information loss values from Experiment 1 (footnote 10, p.9)."""
        dag_d = build_bricofly_detailed(x)
        dag_c = compress_dag(dag_d, 'TC', 'BW', BRICOFLY_COMPRESSION,
                             compressed_cause_name='T')

        loss = compute_information_loss(dag_d, 'TC', dag_c, 'T', 'BW')
        assert abs(loss - expected_loss) < 0.005, \
            f"x={x}: expected loss ~{expected_loss}, got {loss:.4f}"

    def test_loss_monotonic(self):
        """Loss should increase as x diverges from the base rate (0.70)."""
        losses = []
        for x in [0.70, 0.85, 0.98]:
            dag_d = build_bricofly_detailed(x)
            dag_c = compress_dag(dag_d, 'TC', 'BW', BRICOFLY_COMPRESSION,
                                 compressed_cause_name='T')
            losses.append(compute_information_loss(dag_d, 'TC', dag_c, 'T', 'BW'))

        assert losses[0] < losses[1] < losses[2], \
            f"Losses not monotonic: {losses}"


# ---------------------------------------------------------------------------
# Tests: CMI values match paper's footnote 10
# ---------------------------------------------------------------------------

class TestCMIFootnote10:
    """Verify CMI values from footnote 10 (p.9)."""

    def test_marginal_p_blue_wings_x70(self):
        """p(BlueWings) = 0.25[.7 + .7 + .01 + .01] = 0.355."""
        dag = build_bricofly_detailed(0.70)
        joint = dag.compute_joint()
        var_names = list(dag.variables.keys())
        bw_idx = var_names.index('BW')

        p_bw_yes = sum(prob for vals, prob in joint.items() if vals[bw_idx] == 1)
        assert np.isclose(p_bw_yes, 0.355, atol=1e-10)

    def test_marginal_p_blue_wings_x85(self):
        """p(BlueWings) = 0.25[.85 + .7 + .01 + .01] = 0.3925."""
        dag = build_bricofly_detailed(0.85)
        joint = dag.compute_joint()
        var_names = list(dag.variables.keys())
        bw_idx = var_names.index('BW')

        p_bw_yes = sum(prob for vals, prob in joint.items() if vals[bw_idx] == 1)
        assert np.isclose(p_bw_yes, 0.3925, atol=1e-10)

    def test_cmi_less_compressed_x70(self):
        """CMI(4-value TC, BW) ~ 0.47 for x=0.70."""
        dag = build_bricofly_detailed(0.70)
        cmi = compute_cmi(dag, 'TC', 'BW')

        manual_cmi = compute_cmi_manual({
            0: [0.30, 0.70],
            1: [0.30, 0.70],
            2: [0.99, 0.01],
            3: [0.99, 0.01],
        })

        assert np.isclose(cmi, manual_cmi, atol=1e-10)
        assert abs(cmi - 0.47) < 0.02

    def test_cmi_more_compressed_x70(self):
        """CMI(2-value T, BW) ~ 0.47 for x=0.70."""
        dag_d = build_bricofly_detailed(0.70)
        dag_c = compress_dag(dag_d, 'TC', 'BW', BRICOFLY_COMPRESSION,
                             compressed_cause_name='T')
        cmi = compute_cmi(dag_c, 'T', 'BW')

        manual_cmi = compute_cmi_manual({
            0: [0.30, 0.70],
            1: [0.99, 0.01],
        })

        assert np.isclose(cmi, manual_cmi, atol=1e-10)
        assert abs(cmi - 0.47) < 0.02

    def test_information_loss_zero_x70(self):
        """Information loss = 0 for x=0.70."""
        dag_d = build_bricofly_detailed(0.70)
        dag_c = compress_dag(dag_d, 'TC', 'BW', BRICOFLY_COMPRESSION,
                             compressed_cause_name='T')
        loss = compute_information_loss(dag_d, 'TC', dag_c, 'T', 'BW')
        assert np.isclose(loss, 0.0, atol=1e-10)

    def test_cmi_less_compressed_x85(self):
        """CMI(4-value TC, BW) ~ 0.55 for x=0.85."""
        dag = build_bricofly_detailed(0.85)
        cmi = compute_cmi(dag, 'TC', 'BW')

        manual_cmi = compute_cmi_manual({
            0: [0.15, 0.85],
            1: [0.30, 0.70],
            2: [0.99, 0.01],
            3: [0.99, 0.01],
        })

        assert np.isclose(cmi, manual_cmi, atol=1e-10)
        assert abs(cmi - 0.55) < 0.02

    def test_cmi_more_compressed_x85(self):
        """CMI(2-value T, BW) ~ 0.54 for x=0.85."""
        dag_d = build_bricofly_detailed(0.85)
        dag_c = compress_dag(dag_d, 'TC', 'BW', BRICOFLY_COMPRESSION,
                             compressed_cause_name='T')
        cmi = compute_cmi(dag_c, 'T', 'BW')

        manual_cmi = compute_cmi_manual({
            0: [0.225, 0.775],  # avg(0.85, 0.70)
            1: [0.99, 0.01],
        })

        assert np.isclose(cmi, manual_cmi, atol=1e-10)
        assert abs(cmi - 0.54) < 0.02


# ---------------------------------------------------------------------------
# Tests: Experiment 2 loss values
# ---------------------------------------------------------------------------

class TestExperiment2Losses:
    """
    Verify information loss for Experiment 2 parameters (Section 3, p.12-13).

    Experiment 2 replaces the 70% base rate with 55%.
    Loss values from paper (Section 3.2, p.13): 0, 0.04, 0.11.
    """

    @staticmethod
    def build_exp2_detailed(x: float):
        """Experiment 2 DAG: same structure, sentence (b) uses 55%."""
        variables = {
            'TC': Variable('TC', (0, 1, 2, 3)),
            'BW': Variable('BW', (0, 1)),
        }
        parents = {'TC': [], 'BW': ['TC']}

        def cpt_TC(p):
            return np.array([0.25, 0.25, 0.25, 0.25])

        def cpt_BW(p):
            tc = p['TC']
            probs = {0: x, 1: 0.55, 2: 0.01, 3: 0.01}
            p_yes = probs[tc]
            return np.array([1.0 - p_yes, p_yes])

        return CausalDAG(variables, parents, {'TC': cpt_TC, 'BW': cpt_BW})

    @pytest.mark.parametrize("x,expected_loss", [
        (0.55, 0.00),
        (0.85, 0.04),
        (0.98, 0.11),
    ])
    def test_experiment2_losses(self, x, expected_loss):
        dag_d = self.build_exp2_detailed(x)
        dag_c = compress_dag(dag_d, 'TC', 'BW', BRICOFLY_COMPRESSION,
                             compressed_cause_name='T')
        loss = compute_information_loss(dag_d, 'TC', dag_c, 'T', 'BW')
        assert abs(loss - expected_loss) < 0.01, \
            f"x={x}: expected ~{expected_loss}, got {loss:.4f}"


# ---------------------------------------------------------------------------
# Tests: CMI mathematical properties
# ---------------------------------------------------------------------------

class TestCMIProperties:
    """Verify mathematical properties that should always hold."""

    def test_cmi_nonnegative(self):
        """CMI >= 0 (mutual information is non-negative)."""
        for x in [0.01, 0.50, 0.70, 0.85, 0.98, 0.99]:
            dag = build_bricofly_detailed(x)
            cmi = compute_cmi(dag, 'TC', 'BW')
            assert cmi >= -1e-10, f"CMI = {cmi} < 0 for x={x}"

    def test_information_loss_nonnegative(self):
        """Information loss >= 0 (compression cannot increase information)."""
        for x in [0.01, 0.50, 0.70, 0.85, 0.98, 0.99]:
            dag_d = build_bricofly_detailed(x)
            dag_c = compress_dag(dag_d, 'TC', 'BW', BRICOFLY_COMPRESSION,
                                 compressed_cause_name='T')
            loss = compute_information_loss(dag_d, 'TC', dag_c, 'T', 'BW',
                                            compression_map=BRICOFLY_COMPRESSION)
            assert loss >= -1e-10, f"Loss = {loss} < 0 for x={x}"

    def test_cmi_bounded_by_entropy(self):
        """CMI(C,E) <= H(E), the entropy of the effect variable."""
        for x in [0.50, 0.70, 0.85, 0.98]:
            dag = build_bricofly_detailed(x)
            cmi = compute_cmi(dag, 'TC', 'BW')

            p = 0.25 * (x + 0.70 + 0.01 + 0.01)
            h_bw = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

            assert cmi <= h_bw + 1e-10, \
                f"CMI ({cmi}) > H(E) ({h_bw}) for x={x}"

    def test_lossless_compression_zero_loss(self):
        """When probabilities are identical within groups, loss = 0."""
        dag_d = build_bricofly_detailed(0.70)
        dag_c = compress_dag(dag_d, 'TC', 'BW', BRICOFLY_COMPRESSION,
                             compressed_cause_name='T')
        loss = compute_information_loss(dag_d, 'TC', dag_c, 'T', 'BW')
        assert np.isclose(loss, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: compute_cmi_multivar consistency
# ---------------------------------------------------------------------------

class TestCMIMultivarConsistency:
    """Verify that compute_cmi_multivar agrees with compute_cmi for single vars."""

    def test_single_var_equivalence(self):
        """compute_cmi_multivar(['TC'], 'BW') should equal compute_cmi('TC', 'BW')."""
        for x in [0.70, 0.85, 0.98]:
            dag = build_bricofly_detailed(x)
            cmi_single = compute_cmi(dag, 'TC', 'BW')
            cmi_multi = compute_cmi_multivar(dag, ['TC'], 'BW')
            assert np.isclose(cmi_single, cmi_multi, atol=1e-10), \
                f"Single ({cmi_single}) != Multi ({cmi_multi}) for x={x}"


# ---------------------------------------------------------------------------
# Tests: Stability condition (Report 2 from paper)
# ---------------------------------------------------------------------------

class TestStabilityCompression:
    """
    Verify compression via removing a background variable (stability).

    Report 2 in the paper compresses by removing the background condition
    (e.g., water spray vs dry air blow) rather than coarsening the cause.
    The compressed model has Tank Temp only, with BW CPT averaged over
    the removed background variable.
    """

    @staticmethod
    def build_stability_detailed(x: float):
        """
        Less compressed model for stability condition.

        Variables: TankTemp {Warm=0, Cold=1}, Treatment {Water=0, Air=1}, BW {No=0, Yes=1}

        P(BW=1 | Warm, Water) = x
        P(BW=1 | Warm, Air)   = 0.70
        P(BW=1 | Cold, Water) = 0.01
        P(BW=1 | Cold, Air)   = 0.01
        """
        variables = {
            'TT': Variable('TT', (0, 1)),
            'Tr': Variable('Tr', (0, 1)),
            'BW': Variable('BW', (0, 1)),
        }
        parents = {'TT': [], 'Tr': [], 'BW': ['TT', 'Tr']}

        def cpt_TT(p):
            return np.array([0.5, 0.5])

        def cpt_Tr(p):
            return np.array([0.5, 0.5])

        def cpt_BW(p):
            tt, tr = p['TT'], p['Tr']
            probs = {
                (0, 0): x,     # Warm, Water
                (0, 1): 0.70,  # Warm, Air
                (1, 0): 0.01,  # Cold, Water
                (1, 1): 0.01,  # Cold, Air
            }
            p_yes = probs[(tt, tr)]
            return np.array([1.0 - p_yes, p_yes])

        return CausalDAG(variables, parents, {'TT': cpt_TT, 'Tr': cpt_Tr, 'BW': cpt_BW})

    @staticmethod
    def build_stability_compressed(x: float):
        """
        More compressed model: remove Treatment, keep only TankTemp.

        P(BW=1|do(Warm)) = avg over Treatment = (x + 0.70) / 2
        P(BW=1|do(Cold)) = avg over Treatment = (0.01 + 0.01) / 2 = 0.01
        """
        avg_warm = (x + 0.70) / 2.0

        variables = {
            'TT': Variable('TT', (0, 1)),
            'BW': Variable('BW', (0, 1)),
        }
        parents = {'TT': [], 'BW': ['TT']}

        def cpt_TT(p):
            return np.array([0.5, 0.5])

        def cpt_BW(p):
            tt = p['TT']
            if tt == 0:
                return np.array([1.0 - avg_warm, avg_warm])
            else:
                return np.array([0.99, 0.01])

        return CausalDAG(variables, parents, {'TT': cpt_TT, 'BW': cpt_BW})

    @pytest.mark.parametrize("x,expected_loss", [
        (0.70, 0.00),
        (0.85, 0.01),
        (0.98, 0.06),
    ])
    def test_stability_loss_matches_proportionality(self, x, expected_loss):
        """
        The paper finds NO difference between proportionality and stability
        compression in terms of information loss. Both should give the same
        loss values.
        """
        dag_d = self.build_stability_detailed(x)
        dag_c = self.build_stability_compressed(x)

        # For stability, the cause in the detailed model is the joint
        # (TT, Tr), and in the compressed model it's just TT.
        # But the paper measures loss as CMI difference between the
        # 4-value joint cause and the 2-value compressed cause.
        cmi_detailed = compute_cmi_multivar(dag_d, ['TT', 'Tr'], 'BW')
        cmi_compressed = compute_cmi(dag_c, 'TT', 'BW')
        loss = cmi_detailed - cmi_compressed

        assert abs(loss - expected_loss) < 0.005, \
            f"Stability loss x={x}: expected ~{expected_loss}, got {loss:.4f}"


# ---------------------------------------------------------------------------
# Tests: Appendix A simulations (page 31)
# ---------------------------------------------------------------------------

def build_appendix_3val(x, p2, p3):
    """
    Build the 3-value cause DAG from the appendix simulation.

    C = {c1=0, c2=1, c3=2}, E = {e1=0, e2=1}
    p(e1|do(c1)) = x, p(e1|do(c2)) = p2, p(e1|do(c3)) = p3
    Uniform q(c) = 1/3.
    """
    variables = {
        'C': Variable('C', (0, 1, 2)),
        'E': Variable('E', (0, 1)),
    }
    parents = {'C': [], 'E': ['C']}

    def cpt_C(p):
        return np.array([1/3, 1/3, 1/3])

    def cpt_E(p):
        c = p['C']
        probs = {0: x, 1: p2, 2: p3}
        py = probs[c]
        return np.array([1 - py, py])

    return CausalDAG(variables, parents, {'C': cpt_C, 'E': cpt_E})


# Appendix compression: {c1,c2} -> chat1 (0), {c3} -> chat2 (1)
APPENDIX_COMPRESSION = {0: 0, 1: 0, 2: 1}


class TestAppendixSimulation:
    """
    Verify information loss for the appendix simulation (page 31, Figure 9).

    3-value cause compressed to 2 values by merging {c1,c2} -> chat1.
    Uses the induced reference distribution q(chat1)=2/3, q(chat2)=1/3.
    """

    def test_loss_zero_when_merged_values_identical_x05(self):
        """When p(e1|do(c1)) = p(e1|do(c2)) = 0.5, loss = 0 regardless of p3."""
        for p3 in [0.0, 0.25, 0.5, 0.75, 1.0]:
            dag = build_appendix_3val(0.5, 0.5, p3)
            dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                                 compressed_cause_name='Ch')
            loss = compute_information_loss(dag, 'C', dag_c, 'Ch', 'E',
                                           compression_map=APPENDIX_COMPRESSION)
            assert np.isclose(loss, 0.0, atol=1e-10), \
                f"p3={p3}: loss = {loss}, expected 0"

    def test_loss_zero_when_merged_values_identical_x10(self):
        """When p(e1|do(c1)) = p(e1|do(c2)) = 1.0, loss = 0 regardless of p3."""
        for p3 in [0.0, 0.25, 0.5, 0.75, 1.0]:
            dag = build_appendix_3val(1.0, 1.0, p3)
            dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                                 compressed_cause_name='Ch')
            loss = compute_information_loss(dag, 'C', dag_c, 'Ch', 'E',
                                           compression_map=APPENDIX_COMPRESSION)
            assert np.isclose(loss, 0.0, atol=1e-10), \
                f"p3={p3}: loss = {loss}, expected 0"

    def test_loss_nonnegative_sweep(self):
        """Information loss must be >= 0 for all parameter values."""
        for x in [0.0, 0.5, 1.0]:
            for p2 in np.linspace(0.01, 0.99, 5):
                for p3 in np.linspace(0.01, 0.99, 5):
                    dag = build_appendix_3val(x, p2, p3)
                    dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                                         compressed_cause_name='Ch')
                    loss = compute_information_loss(
                        dag, 'C', dag_c, 'Ch', 'E',
                        compression_map=APPENDIX_COMPRESSION)
                    assert loss >= -1e-10, \
                        f"x={x}, p2={p2:.2f}, p3={p3:.2f}: loss={loss}"

    def test_loss_increases_with_divergence_x05(self):
        """Fig 9(a): loss increases as p2 diverges from x=0.5."""
        p3 = 0.5  # fix p3
        losses = []
        for p2 in [0.5, 0.3, 0.1]:
            dag = build_appendix_3val(0.5, p2, p3)
            dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                                 compressed_cause_name='Ch')
            losses.append(compute_information_loss(
                dag, 'C', dag_c, 'Ch', 'E',
                compression_map=APPENDIX_COMPRESSION))
        assert losses[0] < losses[1] < losses[2], \
            f"Loss not monotonically increasing: {losses}"

    def test_loss_increases_with_divergence_x10(self):
        """Fig 9(b): loss increases as p2 diverges from x=1.0."""
        p3 = 0.5
        losses = []
        for p2 in [0.99, 0.7, 0.3]:
            dag = build_appendix_3val(1.0, p2, p3)
            dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                                 compressed_cause_name='Ch')
            losses.append(compute_information_loss(
                dag, 'C', dag_c, 'Ch', 'E',
                compression_map=APPENDIX_COMPRESSION))
        assert losses[0] < losses[1] < losses[2], \
            f"Loss not monotonically increasing: {losses}"

    def test_loss_symmetric_around_x05(self):
        """Fig 9(a): for x=0.5, loss(p2=x+d) = loss(p2=x-d) by symmetry."""
        p3 = 0.3
        for d in [0.1, 0.2, 0.4]:
            dag_hi = build_appendix_3val(0.5, 0.5 + d, p3)
            dag_lo = build_appendix_3val(0.5, 0.5 - d, p3)
            dag_c_hi = compress_dag(dag_hi, 'C', 'E', APPENDIX_COMPRESSION,
                                    compressed_cause_name='Ch')
            dag_c_lo = compress_dag(dag_lo, 'C', 'E', APPENDIX_COMPRESSION,
                                    compressed_cause_name='Ch')
            loss_hi = compute_information_loss(
                dag_hi, 'C', dag_c_hi, 'Ch', 'E',
                compression_map=APPENDIX_COMPRESSION)
            loss_lo = compute_information_loss(
                dag_lo, 'C', dag_c_lo, 'Ch', 'E',
                compression_map=APPENDIX_COMPRESSION)
            assert np.isclose(loss_hi, loss_lo, atol=1e-10), \
                f"d={d}: loss(0.5+d)={loss_hi}, loss(0.5-d)={loss_lo}"

    def test_max_loss_at_extremes_x05(self):
        """Fig 9(a): for x=0.5, maximum loss when p2 is at 0 or 1."""
        p3 = 0.5
        loss_extreme = []
        loss_moderate = []
        for p2 in [0.01, 0.99]:
            dag = build_appendix_3val(0.5, p2, p3)
            dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                                 compressed_cause_name='Ch')
            loss_extreme.append(compute_information_loss(
                dag, 'C', dag_c, 'Ch', 'E',
                compression_map=APPENDIX_COMPRESSION))
        for p2 in [0.3, 0.7]:
            dag = build_appendix_3val(0.5, p2, p3)
            dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                                 compressed_cause_name='Ch')
            loss_moderate.append(compute_information_loss(
                dag, 'C', dag_c, 'Ch', 'E',
                compression_map=APPENDIX_COMPRESSION))
        assert all(le > lm for le in loss_extreme for lm in loss_moderate), \
            f"Extreme losses {loss_extreme} not > moderate losses {loss_moderate}"

    def test_regression_induced_q_required(self):
        """Regression: without induced q, unequal groups give negative loss."""
        dag = build_appendix_3val(0.5, 0.5, 0.0)
        dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                             compressed_cause_name='Ch')
        # Without compression_map, uses uniform q -- gives wrong negative loss
        loss_wrong = compute_information_loss(dag, 'C', dag_c, 'Ch', 'E')
        assert loss_wrong < 0, \
            "Expected negative loss without induced q (regression baseline)"
        # With compression_map, uses induced q -- gives correct zero loss
        loss_correct = compute_information_loss(
            dag, 'C', dag_c, 'Ch', 'E',
            compression_map=APPENDIX_COMPRESSION)
        assert np.isclose(loss_correct, 0.0, atol=1e-10), \
            f"Expected zero loss with induced q, got {loss_correct}"


# ---------------------------------------------------------------------------
# Tests: CMI decomposition as sum of weighted KL divergences (Eq 1, p.6)
# ---------------------------------------------------------------------------

def _kl_divergence(p, q):
    """DKL(p || q) in bits for two distributions (arrays)."""
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0 and qi > 0:
            kl += pi * np.log2(pi / qi)
    return kl


def _interventional_dists(dag, cause_var, effect_var):
    """Extract p(e|do(c)) for each c as a dict of arrays."""
    cause = dag.variables[cause_var]
    effect = dag.variables[effect_var]
    var_names = list(dag.variables.keys())
    e_idx = var_names.index(effect_var)

    dists = {}
    for c in cause.domain:
        joint = dag.compute_joint(interventions={cause_var: c})
        p_e = np.zeros(len(effect.domain))
        for vals, prob in joint.items():
            e_val_idx = effect.domain.index(vals[e_idx])
            p_e[e_val_idx] += prob
        dists[c] = p_e
    return dists


class TestCMIDecomposition:
    """
    CMI(C, E) = sum_c q(c) * DKL(p(e|do(c)) || p(e))

    Verify the decomposition explicitly by computing each DKL term
    independently and comparing the sum against compute_cmi.
    """

    @pytest.mark.parametrize("x", [0.50, 0.70, 0.85, 0.98])
    def test_cmi_equals_weighted_kl_sum_bricofly(self, x):
        dag = build_bricofly_detailed(x)
        cause = dag.variables['TC']
        n_c = len(cause.domain)
        q = 1.0 / n_c

        dists = _interventional_dists(dag, 'TC', 'BW')

        # Marginal: p(e) = sum_c q(c) * p(e|do(c))
        p_marginal = sum(q * dists[c] for c in cause.domain)

        # CMI via decomposition
        cmi_decomposed = sum(
            q * _kl_divergence(dists[c], p_marginal) for c in cause.domain
        )

        cmi_function = compute_cmi(dag, 'TC', 'BW')
        assert np.isclose(cmi_decomposed, cmi_function, atol=1e-10), \
            f"x={x}: decomposed={cmi_decomposed}, function={cmi_function}"

    def test_cmi_equals_weighted_kl_sum_3val(self):
        """Also verify for the 3-value appendix DAG."""
        for x, p2, p3 in [(0.1, 0.5, 0.9), (0.5, 0.3, 0.7), (0.8, 0.2, 0.6)]:
            dag = build_appendix_3val(x, p2, p3)
            cause = dag.variables['C']
            q = 1.0 / 3.0

            dists = _interventional_dists(dag, 'C', 'E')
            p_marginal = sum(q * dists[c] for c in cause.domain)

            cmi_decomposed = sum(
                q * _kl_divergence(dists[c], p_marginal) for c in cause.domain
            )
            cmi_function = compute_cmi(dag, 'C', 'E')
            assert np.isclose(cmi_decomposed, cmi_function, atol=1e-10), \
                f"({x},{p2},{p3}): decomposed={cmi_decomposed}, function={cmi_function}"


# ---------------------------------------------------------------------------
# Tests: CMI bounded by log2|C| (cause entropy under uniform q)
# ---------------------------------------------------------------------------

class TestCMIUpperBound:
    """
    CMI(C, E) <= H(q) = log2|C| when q is uniform.

    This is the standard mutual information bound: I(X;Y) <= H(X).
    """

    @pytest.mark.parametrize("x", [0.01, 0.50, 0.70, 0.85, 0.98, 0.99])
    def test_cmi_bounded_by_cause_entropy_bricofly(self, x):
        dag = build_bricofly_detailed(x)
        cmi = compute_cmi(dag, 'TC', 'BW')
        h_cause = np.log2(4)  # 4-value TC, uniform q -> H(q) = 2 bits
        assert cmi <= h_cause + 1e-10, \
            f"x={x}: CMI={cmi} > log2(4)={h_cause}"

    def test_cmi_bounded_by_cause_entropy_3val(self):
        for x, p2, p3 in [(0.1, 0.5, 0.9), (0.0, 0.5, 1.0)]:
            dag = build_appendix_3val(x, p2, p3)
            cmi = compute_cmi(dag, 'C', 'E')
            h_cause = np.log2(3)
            assert cmi <= h_cause + 1e-10, \
                f"({x},{p2},{p3}): CMI={cmi} > log2(3)={h_cause}"

    def test_cmi_bounded_by_both_entropies(self):
        """CMI <= min(H(q), H(E)): tighter than either bound alone."""
        for x in [0.50, 0.85, 0.98]:
            dag = build_bricofly_detailed(x)
            cmi = compute_cmi(dag, 'TC', 'BW')

            h_cause = np.log2(4)
            p = 0.25 * (x + 0.70 + 0.01 + 0.01)
            h_effect = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

            assert cmi <= min(h_cause, h_effect) + 1e-10


# ---------------------------------------------------------------------------
# Tests: Information loss as within-group KL divergence
# ---------------------------------------------------------------------------

class TestLossWithinGroupKL:
    """
    Information loss can be decomposed as a weighted sum of within-group
    KL divergences:

        L = sum_c' q(c') * sum_{c in sigma^{-1}(c')} [q(c)/q(c')]
            * DKL(p(e|do(c)) || p(e|do(c')))

    This makes L >= 0 obvious (KL is non-negative) and provides an
    independent computation path to verify compute_information_loss.
    """

    @pytest.mark.parametrize("x", [0.70, 0.85, 0.98])
    def test_within_group_kl_matches_loss_bricofly(self, x):
        """Verify decomposition for 4->2 Bricofly compression."""
        dag = build_bricofly_detailed(x)
        dag_c = compress_dag(dag, 'TC', 'BW', BRICOFLY_COMPRESSION,
                             compressed_cause_name='T')

        # Get interventional distributions
        dists = _interventional_dists(dag, 'TC', 'BW')
        dists_compressed = _interventional_dists(dag_c, 'T', 'BW')

        # Build inverse map
        inverse = {}
        for orig, comp in BRICOFLY_COMPRESSION.items():
            inverse.setdefault(comp, []).append(orig)

        q_c = 0.25  # uniform over 4 values

        # L = sum_c' q(c') * sum_{c in group} [q(c)/q(c')] * DKL(p_c || p_c')
        loss_decomposed = 0.0
        for c_hat, originals in inverse.items():
            q_c_hat = q_c * len(originals)
            for c in originals:
                loss_decomposed += q_c * _kl_divergence(
                    dists[c], dists_compressed[c_hat]
                )

        loss_function = compute_information_loss(dag, 'TC', dag_c, 'T', 'BW')
        assert np.isclose(loss_decomposed, loss_function, atol=1e-10), \
            f"x={x}: within-group={loss_decomposed}, function={loss_function}"

    def test_within_group_kl_matches_loss_3val(self):
        """Verify decomposition for 3->2 appendix compression."""
        for x, p2, p3 in [(0.5, 0.3, 0.7), (1.0, 0.5, 0.2), (0.1, 0.9, 0.5)]:
            dag = build_appendix_3val(x, p2, p3)
            dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                                 compressed_cause_name='Ch')

            dists = _interventional_dists(dag, 'C', 'E')
            dists_compressed = _interventional_dists(dag_c, 'Ch', 'E')

            inverse = {}
            for orig, comp in APPENDIX_COMPRESSION.items():
                inverse.setdefault(comp, []).append(orig)

            q_c = 1.0 / 3.0

            loss_decomposed = 0.0
            for c_hat, originals in inverse.items():
                for c in originals:
                    loss_decomposed += q_c * _kl_divergence(
                        dists[c], dists_compressed[c_hat]
                    )

            loss_function = compute_information_loss(
                dag, 'C', dag_c, 'Ch', 'E',
                compression_map=APPENDIX_COMPRESSION)
            assert np.isclose(loss_decomposed, loss_function, atol=1e-10), \
                f"({x},{p2},{p3}): within-group={loss_decomposed}, function={loss_function}"


# ---------------------------------------------------------------------------
# Tests: Loss independent of unmerged value (Fig 9, p.31)
# ---------------------------------------------------------------------------

class TestLossIndependentOfUnmerged:
    """
    For compression {c1,c2} -> chat1, {c3} -> chat2, with induced q,
    the information loss depends only on p(e|do(c1)) and p(e|do(c2)).
    It is exactly independent of p(e|do(c3)).

    Proof: the DKL(p3 || p_avg) terms in CMI_detailed and CMI_compressed
    cancel exactly when using induced q.
    """

    @pytest.mark.parametrize("x,p2", [
        (0.5, 0.3), (0.5, 0.8), (1.0, 0.5), (0.2, 0.7),
    ])
    def test_loss_invariant_to_p3(self, x, p2):
        """Loss should be identical for any p3 value."""
        p3_values = [0.01, 0.25, 0.5, 0.75, 0.99]
        losses = []
        for p3 in p3_values:
            dag = build_appendix_3val(x, p2, p3)
            dag_c = compress_dag(dag, 'C', 'E', APPENDIX_COMPRESSION,
                                 compressed_cause_name='Ch')
            loss = compute_information_loss(
                dag, 'C', dag_c, 'Ch', 'E',
                compression_map=APPENDIX_COMPRESSION)
            losses.append(loss)

        for i in range(1, len(losses)):
            assert np.isclose(losses[0], losses[i], atol=1e-10), \
                f"x={x}, p2={p2}: loss varies with p3: {losses}"


# ---------------------------------------------------------------------------
# Tests: Additivity of loss for chained compressions
# ---------------------------------------------------------------------------

class TestLossAdditivity:
    """
    For chained compressions C -> C' -> C'':
        L(C, C'', E) = L(C, C', E) + L(C', C'', E)

    This follows from L being a CMI difference:
        L(C,C'',E) = CMI(C,E) - CMI(C'',E)
                    = [CMI(C,E) - CMI(C',E)] + [CMI(C',E) - CMI(C'',E)]
    """

    def test_chained_4_to_3_to_2(self):
        """Compress 4 values in two stages and verify additivity."""
        dag_4 = build_bricofly_detailed(0.85)

        # Stage 1: merge {WH, WD} -> Warm, keep CH, CD separate
        # {0,1} -> 0, {2} -> 1, {3} -> 2
        compression_4_to_3 = {0: 0, 1: 0, 2: 1, 3: 2}
        dag_3 = compress_dag(dag_4, 'TC', 'BW', compression_4_to_3,
                             compressed_cause_name='TC3')

        # Stage 2: merge {CH, CD} -> Cold
        # {0} -> 0, {1} -> 1, {2} -> 1
        compression_3_to_2 = {0: 0, 1: 1, 2: 1}
        dag_2 = compress_dag(dag_3, 'TC3', 'BW', compression_3_to_2,
                             compressed_cause_name='TC2')

        # Direct: 4 -> 2 (same as BRICOFLY_COMPRESSION)
        dag_2_direct = compress_dag(dag_4, 'TC', 'BW', BRICOFLY_COMPRESSION,
                                    compressed_cause_name='T')

        # Compute losses
        loss_4_to_3 = compute_information_loss(
            dag_4, 'TC', dag_3, 'TC3', 'BW',
            compression_map=compression_4_to_3)
        loss_3_to_2 = compute_information_loss(
            dag_3, 'TC3', dag_2, 'TC2', 'BW',
            compression_map=compression_3_to_2)
        loss_4_to_2 = compute_information_loss(
            dag_4, 'TC', dag_2_direct, 'T', 'BW',
            compression_map=BRICOFLY_COMPRESSION)

        assert np.isclose(loss_4_to_2, loss_4_to_3 + loss_3_to_2, atol=1e-10), \
            f"L(4->2)={loss_4_to_2} != L(4->3)+L(3->2)={loss_4_to_3}+{loss_3_to_2}"

    def test_chained_identity_step_adds_zero(self):
        """An identity compression step should add zero loss."""
        dag_4 = build_bricofly_detailed(0.98)

        # Identity: {0->0, 1->1, 2->2, 3->3}
        identity = {0: 0, 1: 1, 2: 2, 3: 3}
        dag_4_copy = compress_dag(dag_4, 'TC', 'BW', identity,
                                  compressed_cause_name='TC2')

        loss_identity = compute_information_loss(
            dag_4, 'TC', dag_4_copy, 'TC2', 'BW',
            compression_map=identity)

        assert np.isclose(loss_identity, 0.0, atol=1e-10)

        # Then compress: total loss should equal direct loss
        dag_2 = compress_dag(dag_4_copy, 'TC2', 'BW', BRICOFLY_COMPRESSION,
                             compressed_cause_name='T')
        loss_to_2 = compute_information_loss(
            dag_4_copy, 'TC2', dag_2, 'T', 'BW',
            compression_map=BRICOFLY_COMPRESSION)
        loss_direct = compute_information_loss(
            dag_4, 'TC', compress_dag(dag_4, 'TC', 'BW', BRICOFLY_COMPRESSION,
                                      compressed_cause_name='T'),
            'T', 'BW', compression_map=BRICOFLY_COMPRESSION)

        assert np.isclose(loss_to_2, loss_direct, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Figure 10 qualitative claim (p.31-32)
# ---------------------------------------------------------------------------

class TestFigure10QWeighting:
    """
    From p.31-32: 'the more that the probability density of q is
    concentrated on the causal values c1 and c2 that are put together
    during the process of compression, the greater the overall amount
    of causal information that is lost in compression.'

    Uses the parametric formula to test with non-uniform q.
    """

    @staticmethod
    def _compute_loss_parametric(p1, p2, p3, q1, q2, q3):
        """Compute information loss for compression {c1,c2}->chat1, {c3}->chat2."""
        p_e_do = {
            0: np.array([1 - p1, p1]),
            1: np.array([1 - p2, p2]),
            2: np.array([1 - p3, p3]),
        }
        q = {0: q1, 1: q2, 2: q3}
        p_marginal = q1 * p_e_do[0] + q2 * p_e_do[1] + q3 * p_e_do[2]

        cmi_detailed = 0.0
        for c in [0, 1, 2]:
            for ei in [0, 1]:
                if p_e_do[c][ei] > 0 and p_marginal[ei] > 0:
                    cmi_detailed += q[c] * p_e_do[c][ei] * np.log2(
                        p_e_do[c][ei] / p_marginal[ei])

        q_chat1 = q1 + q2
        q_chat2 = q3
        if q_chat1 > 0:
            p_e_do_chat1 = (q1 * p_e_do[0] + q2 * p_e_do[1]) / q_chat1
        else:
            p_e_do_chat1 = np.array([0.5, 0.5])
        p_e_do_chat2 = p_e_do[2]

        cmi_compressed = 0.0
        for q_ch, p_ch in [(q_chat1, p_e_do_chat1), (q_chat2, p_e_do_chat2)]:
            for ei in [0, 1]:
                if q_ch > 0 and p_ch[ei] > 0 and p_marginal[ei] > 0:
                    cmi_compressed += q_ch * p_ch[ei] * np.log2(
                        p_ch[ei] / p_marginal[ei])

        return max(cmi_detailed - cmi_compressed, 0.0)

    def test_more_weight_on_merged_values_more_loss(self):
        """Increasing q(c1)+q(c2) should increase loss (for fixed p values)."""
        p1, p2, p3 = 0.1, 0.5, 0.9

        # Shift weight from c3 toward c1 and c2
        configs = [
            (0.1, 0.1, 0.8),   # mostly on c3 (unmerged)
            (0.2, 0.2, 0.6),
            (1/3, 1/3, 1/3),   # uniform
            (0.4, 0.4, 0.2),
            (0.45, 0.45, 0.1), # mostly on c1,c2 (merged)
        ]

        losses = [self._compute_loss_parametric(p1, p2, p3, *q) for q in configs]

        for i in range(len(losses) - 1):
            assert losses[i] < losses[i + 1] + 1e-10, \
                f"Loss not monotonically increasing with merged weight: {list(zip(configs, losses))}"

    def test_symmetric_q_gives_max_loss(self):
        """When q is concentrated equally on c1 and c2, loss is highest."""
        p1, p2, p3 = 0.1, 0.5, 0.9

        loss_merged_heavy = self._compute_loss_parametric(p1, p2, p3, 0.45, 0.45, 0.1)
        loss_unmerged_heavy = self._compute_loss_parametric(p1, p2, p3, 0.1, 0.1, 0.8)

        assert loss_merged_heavy > loss_unmerged_heavy

    def test_dag_matches_parametric(self):
        """Cross-check: DAG-based computation with non-uniform q matches parametric formula."""
        p1, p2, p3 = 0.1, 0.5, 0.9

        # Build a 3-valued cause DAG: C -> E
        variables = {
            'C': Variable('C', (0, 1, 2)),
            'E': Variable('E', (0, 1)),
        }
        parents = {'C': [], 'E': ['C']}

        def cpt_C(p):
            return np.array([1/3, 1/3, 1/3])

        def cpt_E(p):
            probs = {0: p1, 1: p2, 2: p3}
            pe = probs[p['C']]
            return np.array([1 - pe, pe])

        dag = CausalDAG(variables, parents, {'C': cpt_C, 'E': cpt_E})
        compression_map = {0: 0, 1: 0, 2: 1}

        for q1, q2, q3 in [(0.1, 0.1, 0.8), (1/3, 1/3, 1/3), (0.45, 0.45, 0.1)]:
            q_weights = {0: q1, 1: q2, 2: q3}

            # Parametric
            loss_param = self._compute_loss_parametric(p1, p2, p3, q1, q2, q3)

            # DAG-based
            cmi_detailed = compute_cmi(dag, 'C', 'E', q=q_weights)
            dag_c = compress_dag(dag, 'C', 'E', compression_map,
                                 compressed_cause_name='C_hat', q=q_weights)
            q_compressed = {}
            for orig, comp in compression_map.items():
                q_compressed[comp] = q_compressed.get(comp, 0.0) + q_weights[orig]
            cmi_compressed = compute_cmi(dag_c, 'C_hat', 'E', q=q_compressed)
            loss_dag = cmi_detailed - cmi_compressed

            assert abs(loss_param - loss_dag) < 1e-10, \
                f"q={q_weights}: parametric={loss_param:.6f}, dag={loss_dag:.6f}"


# ---------------------------------------------------------------------------
# Tests: do-calculus correctness (Appendix A, p.30)
# ---------------------------------------------------------------------------

class TestDoCalculus:
    """
    Verify that compute_joint(interventions={C: c}) correctly implements
    the truncated factorization from do-calculus:

        p(v | do(x)) = product over V not in X of p(v | parents(V))
                       * delta(V=x for V in X)

    Key properties:
    - Intervened variable becomes a point mass
    - Effect distribution depends only on CPT, not on cause's prior
    - Changing the cause's prior should not affect interventional distributions
    """

    def test_intervention_removes_prior(self):
        """p(e|do(c)) should be independent of the prior on C."""
        # Build two DAGs with different priors on TC but same CPTs
        variables = {
            'TC': Variable('TC', (0, 1, 2, 3)),
            'BW': Variable('BW', (0, 1)),
        }
        parents = {'TC': [], 'BW': ['TC']}

        def cpt_BW(p):
            tc = p['TC']
            probs = {0: 0.85, 1: 0.70, 2: 0.01, 3: 0.01}
            return np.array([1 - probs[tc], probs[tc]])

        # Uniform prior
        dag_uniform = CausalDAG(variables, parents, {
            'TC': lambda p: np.array([0.25, 0.25, 0.25, 0.25]),
            'BW': cpt_BW,
        })

        # Skewed prior
        dag_skewed = CausalDAG(variables, parents, {
            'TC': lambda p: np.array([0.7, 0.1, 0.1, 0.1]),
            'BW': cpt_BW,
        })

        # Interventional distributions should be identical
        for tc_val in [0, 1, 2, 3]:
            joint_u = dag_uniform.compute_joint(interventions={'TC': tc_val})
            joint_s = dag_skewed.compute_joint(interventions={'TC': tc_val})

            var_names = list(variables.keys())
            bw_idx = var_names.index('BW')

            p_bw1_u = sum(p for v, p in joint_u.items() if v[bw_idx] == 1)
            p_bw1_s = sum(p for v, p in joint_s.items() if v[bw_idx] == 1)

            assert np.isclose(p_bw1_u, p_bw1_s, atol=1e-10), \
                f"TC={tc_val}: uniform={p_bw1_u}, skewed={p_bw1_s}"

    def test_intervention_is_point_mass(self):
        """do(C=c) should set P(C=c) = 1, P(C=c') = 0 for c' != c."""
        dag = build_bricofly_detailed(0.85)
        var_names = list(dag.variables.keys())
        tc_idx = var_names.index('TC')

        for tc_val in [0, 1, 2, 3]:
            joint = dag.compute_joint(interventions={'TC': tc_val})
            for vals, prob in joint.items():
                if vals[tc_idx] != tc_val:
                    assert prob == 0.0, \
                        f"do(TC={tc_val}): P(TC={vals[tc_idx]})={prob} != 0"

            # Marginal over TC should be a point mass
            p_tc = sum(p for v, p in joint.items() if v[tc_idx] == tc_val)
            assert np.isclose(p_tc, 1.0, atol=1e-10)

    def test_interventional_matches_cpt(self):
        """p(e|do(c)) should exactly match the CPT entry for that c."""
        x = 0.85
        dag = build_bricofly_detailed(x)
        var_names = list(dag.variables.keys())
        bw_idx = var_names.index('BW')

        expected = {0: x, 1: 0.70, 2: 0.01, 3: 0.01}
        for tc_val, p_bw1_expected in expected.items():
            joint = dag.compute_joint(interventions={'TC': tc_val})
            p_bw1 = sum(p for v, p in joint.items() if v[bw_idx] == 1)
            assert np.isclose(p_bw1, p_bw1_expected, atol=1e-10), \
                f"TC={tc_val}: got {p_bw1}, expected {p_bw1_expected}"

    def test_cmi_independent_of_prior(self):
        """CMI should depend only on p(e|do(c)) and q, not the DAG's prior on C."""
        variables = {
            'C': Variable('C', (0, 1)),
            'E': Variable('E', (0, 1)),
        }
        parents = {'C': [], 'E': ['C']}

        def cpt_E(p):
            if p['C'] == 0:
                return np.array([0.2, 0.8])
            else:
                return np.array([0.9, 0.1])

        dag1 = CausalDAG(variables, parents, {
            'C': lambda p: np.array([0.5, 0.5]),
            'E': cpt_E,
        })
        dag2 = CausalDAG(variables, parents, {
            'C': lambda p: np.array([0.9, 0.1]),
            'E': cpt_E,
        })

        # CMI uses q (default uniform), not the DAG's prior on C
        cmi1 = compute_cmi(dag1, 'C', 'E')
        cmi2 = compute_cmi(dag2, 'C', 'E')
        assert np.isclose(cmi1, cmi2, atol=1e-10), \
            f"CMI differs with prior: {cmi1} vs {cmi2}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
