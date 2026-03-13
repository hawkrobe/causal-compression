"""
Reproduce Figures 9 and 10 from Kinney & Lombrozo (2024) Appendix A.

Figure 9: Information loss heatmaps for 3-value cause compressed to 2 values.
    Panel (a): x = p(e1|do(c1)) = 0.5
    Panel (b): x = p(e1|do(c1)) = 1.0
    Axes: p(e1|do(c2)) vs p(e1|do(c3))
    Compression: {c1,c2} -> chat1, {c3} -> chat2
    Uniform q(c) = 1/3

Figure 10: Information loss on the simplex over q distributions.
    Fixed: p(e1|do(c1)) = 0.1, p(e1|do(c2)) = 0.5, p(e1|do(c3)) = 0.9
    Vary q = (q(c1), q(c2), q(c3)) over the probability simplex.
    Vertex layout (matching paper):
        Top = q(c1), Bottom-left = q(c2), Bottom-right = q(c3)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Ensure repo root is on sys.path so `models` is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Output directory for generated figures
_OUTPUT_DIR = _REPO_ROOT / 'figures'


# ---------------------------------------------------------------------------
# Core computation: information loss with arbitrary q
# ---------------------------------------------------------------------------

def compute_loss_parametric(p1, p2, p3, q1, q2, q3):
    """
    Compute information loss for compressing {c1,c2} -> chat1, {c3} -> chat2.

    Args:
        p1, p2, p3: p(e1|do(c1)), p(e1|do(c2)), p(e1|do(c3))
        q1, q2, q3: reference distribution weights over c1, c2, c3

    Returns:
        Information loss L = CMI_detailed - CMI_compressed
    """
    # Interventional distributions: p(e|do(c)) for binary E
    p_e_do = {
        0: np.array([1 - p1, p1]),
        1: np.array([1 - p2, p2]),
        2: np.array([1 - p3, p3]),
    }
    q = {0: q1, 1: q2, 2: q3}

    # Marginal: p(e) = sum_c q(c) * p(e|do(c))
    p_marginal = q1 * p_e_do[0] + q2 * p_e_do[1] + q3 * p_e_do[2]

    # CMI_detailed = sum_c q(c) * DKL(p(e|do(c)) || p_marginal)
    cmi_detailed = 0.0
    for c in [0, 1, 2]:
        for ei in [0, 1]:
            if p_e_do[c][ei] > 0 and p_marginal[ei] > 0:
                cmi_detailed += q[c] * p_e_do[c][ei] * np.log2(
                    p_e_do[c][ei] / p_marginal[ei]
                )

    # Compressed: {c1,c2} -> chat1, {c3} -> chat2
    q_chat1 = q1 + q2
    q_chat2 = q3

    if q_chat1 > 0:
        p_e_do_chat1 = (q1 * p_e_do[0] + q2 * p_e_do[1]) / q_chat1
    else:
        p_e_do_chat1 = np.array([0.5, 0.5])

    p_e_do_chat2 = p_e_do[2]

    # CMI_compressed = sum_chat q(chat) * DKL(p(e|do(chat)) || p_marginal)
    cmi_compressed = 0.0
    for q_ch, p_ch in [(q_chat1, p_e_do_chat1), (q_chat2, p_e_do_chat2)]:
        for ei in [0, 1]:
            if q_ch > 0 and p_ch[ei] > 0 and p_marginal[ei] > 0:
                cmi_compressed += q_ch * p_ch[ei] * np.log2(
                    p_ch[ei] / p_marginal[ei]
                )

    return max(cmi_detailed - cmi_compressed, 0.0)


# ---------------------------------------------------------------------------
# Cross-check: verify parametric computation matches DAG-based computation
# ---------------------------------------------------------------------------

def cross_check():
    """Verify parametric function matches our DAG-based code."""
    from models import (
        Variable, CausalDAG, compute_cmi, compress_dag,
        compute_information_loss,
    )

    comp_map = {0: 0, 1: 0, 2: 1}

    for x, p2, p3 in [(0.5, 0.3, 0.7), (1.0, 0.5, 0.2), (0.5, 0.5, 0.0)]:
        # Parametric
        loss_param = compute_loss_parametric(x, p2, p3, 1/3, 1/3, 1/3)

        # DAG-based
        variables = {
            'C': Variable('C', (0, 1, 2)),
            'E': Variable('E', (0, 1)),
        }
        parents = {'C': [], 'E': ['C']}

        def make_cpt(x_, p2_, p3_):
            def cpt_E(p):
                c = p['C']
                return np.array([1 - [x_, p2_, p3_][c], [x_, p2_, p3_][c]])
            return cpt_E

        dag = CausalDAG(
            variables, parents,
            {'C': lambda p: np.array([1/3, 1/3, 1/3]),
             'E': make_cpt(x, p2, p3)},
        )
        dag_c = compress_dag(dag, 'C', 'E', comp_map,
                             compressed_cause_name='Ch')
        loss_dag = compute_information_loss(
            dag, 'C', dag_c, 'Ch', 'E', compression_map=comp_map,
        )

        assert np.isclose(loss_param, loss_dag, atol=1e-10), \
            f"Mismatch: param={loss_param}, dag={loss_dag} for ({x},{p2},{p3})"

    print("Cross-check passed: parametric matches DAG-based computation.")


# ---------------------------------------------------------------------------
# Figure 9: Information Loss By Parameter
# ---------------------------------------------------------------------------

def make_figure_9():
    """Reproduce Figure 9: heatmaps for x=0.5 and x=1.0.

    NOTE: The paper's figure has the axis labels swapped. The loss is
    mathematically independent of p(e1|do(c3)) (provably: the DKL(p3||p_avg)
    terms cancel). The gradient runs across p(e1|do(c2)), the merged value.
    We swap axes here to match the paper's visual layout.
    """
    n = 200
    p2_vals = np.linspace(0.001, 0.999, n)
    p3_vals = np.linspace(0.001, 0.999, n)
    P2, P3 = np.meshgrid(p2_vals, p3_vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (x, title_x) in enumerate([(0.5, 'x=.5'), (1.0, 'x=1')]):
        loss = np.zeros_like(P2)
        for i in range(n):
            for j in range(n):
                loss[i, j] = compute_loss_parametric(
                    x, P2[i, j], P3[i, j], 1/3, 1/3, 1/3
                )

        ax = axes[ax_idx]
        # Swap axes to match paper's visual layout (paper has labels swapped)
        # x-axis: p3 (irrelevant variable), y-axis: p2 (merged variable)
        im = ax.pcolormesh(p3_vals, p2_vals, loss.T, cmap='gray_r',
                           shading='auto')
        fig.colorbar(im, ax=ax, label='Information Loss')
        ax.set_xlabel(r'p(e$_1$|do(c$_2$))')
        ax.set_ylabel(r'p(e$_1$|do(c$_3$))')
        ax.set_title(f'Information Loss By Parameter ({title_x})')
        ax.set_aspect('equal')

        label = '(a)' if ax_idx == 0 else '(b)'
        ax.text(0.5, -0.12, label, transform=ax.transAxes,
                ha='center', fontsize=14)

    plt.tight_layout()
    out = _OUTPUT_DIR / 'figure_9_reproduction.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 10: Information Loss on the q-simplex
# ---------------------------------------------------------------------------

def make_figure_10():
    """Reproduce Figure 10: ternary contour plot over q distributions.

    Paper vertex layout:
        Top      = q(c1) = 1
        Bot-left = q(c2) = 1
        Bot-right= q(c3) = 1

    Cartesian mapping (top-pointing equilateral triangle):
        q(c1) at top:       (0.5, sqrt(3)/2)
        q(c2) at bot-left:  (0, 0)
        q(c3) at bot-right: (1, 0)

        x = q3 + q1/2
        y = q1 * sqrt(3)/2
    """
    p1, p2, p3 = 0.1, 0.5, 0.9

    # Generate points on the simplex
    n = 150
    points = []
    values = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            q1 = i / n
            q2 = j / n
            q3 = k / n
            if q1 < 0.005 or q2 < 0.005 or q3 < 0.005:
                continue
            loss = compute_loss_parametric(p1, p2, p3, q1, q2, q3)
            points.append((q1, q2, q3))
            values.append(loss)

    points = np.array(points)
    values = np.array(values)

    # Convert to Cartesian: q(c1) at top, q(c2) at bot-left, q(c3) at bot-right
    x_cart = points[:, 2] + points[:, 0] / 2   # q3 + q1/2
    y_cart = points[:, 0] * np.sqrt(3) / 2     # q1 * sqrt(3)/2

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Triangulate
    triang = tri.Triangulation(x_cart, y_cart)

    # Filled contours (matching paper's banded style)
    levels = np.linspace(0, 0.16, 9)
    tcf = ax.tricontourf(triang, values, levels=levels, cmap='gray_r')
    # Contour lines
    ax.tricontour(triang, values, levels=levels, colors='gray',
                  linewidths=0.5)

    cbar = fig.colorbar(tcf, ax=ax, label='Information Loss')

    # Triangle boundary
    s3 = np.sqrt(3) / 2
    corners_x = [0, 1, 0.5, 0]
    corners_y = [0, 0, s3, 0]
    ax.plot(corners_x, corners_y, 'k-', linewidth=1.5)

    # Edge tick marks matching paper layout
    for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        # Left edge: q(c2) axis -- q(c2) increases from top (0) to bot-left (1)
        # Position along left edge: from top (0.5, s3) to bot-left (0, 0)
        lx = 0.5 * (1 - frac)
        ly = s3 * (1 - frac)
        ax.text(lx - 0.03, ly + 0.01, f'{frac:.1f}', ha='right',
                va='center', fontsize=7, rotation=60)

        # Right edge: q(c1) axis -- q(c1) increases from bot-right (0) to top (1)
        # Position along right edge: from bot-right (1, 0) to top (0.5, s3)
        rx = 1 - 0.5 * frac
        ry = s3 * frac
        ax.text(rx + 0.03, ry + 0.01, f'{frac:.1f}', ha='left',
                va='center', fontsize=7, rotation=-60)

        # Bottom edge: q(c3) axis -- q(c3) increases from bot-left (0) to bot-right (1)
        bx = frac
        ax.text(bx, -0.04, f'{frac:.1f}', ha='center', va='top',
                fontsize=7)

    # Edge axis labels (rotated, matching paper)
    # Left edge midpoint
    ax.text(0.15, s3 * 0.55, r'q(c$_2$)', ha='center', va='center',
            fontsize=11, rotation=60)
    # Right edge midpoint
    ax.text(0.85, s3 * 0.55, r'q(c$_1$)', ha='center', va='center',
            fontsize=11, rotation=-60)
    # Bottom edge
    ax.text(0.5, -0.09, r'q(c$_3$)', ha='center', va='top', fontsize=11)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, s3 + 0.12)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    out = _OUTPUT_DIR / 'figure_10_reproduction.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    _OUTPUT_DIR.mkdir(exist_ok=True)
    cross_check()
    print()
    print("Generating Figure 9...")
    make_figure_9()
    print("Generating Figure 10...")
    make_figure_10()
    print()
    print("Done. Compare with paper's Figures 9 and 10.")
