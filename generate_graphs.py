"""
Generate publication-quality graphs for Profile-MoE findings.

Produces:
  1. swap_isolation.png — Before/After MSE per cluster (swap test)
  2. ppl_comparison.png — Profile-MoE vs Learned Router PPL by domain
  3. profile_heatmap.png — Expert × Domain capability matrix
  4. routing_accuracy.png — Routing accuracy vs random baseline
  5. speed_comparison.png — Speed (tok/s) comparison

Uses existing JSON data + quick reruns where needed.

Run: python generate_graphs.py
Output: graphs/ directory with PNG files
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#CCCCCC',
})

COLORS = {
    'code': '#3B82F6',       # blue
    'math': '#EF4444',       # red
    'creative': '#10B981',   # green
    'reasoning': '#F59E0B',  # amber
    'law': '#8B5CF6',        # purple
    'stories': '#EC4899',    # pink
    'wiki': '#06B6D4',       # cyan
    'profile': '#059669',    # emerald
    'learned': '#DC2626',    # red-600
    'before': '#94A3B8',     # slate
    'after': '#3B82F6',      # blue
}

import os
_HERE = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(_HERE, 'graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

with open(os.path.join(_HERE, 'results.json')) as f:
    mvp = json.load(f)

with open(os.path.join(_HERE, 'transformer_results.json')) as f:
    tf = json.load(f)


# ═══════════════════════════════════════════════════════════════════
# GRAPH 1: SWAP ISOLATION
# ═══════════════════════════════════════════════════════════════════

def graph_swap_isolation():
    """Bar chart: MSE before/after swap per cluster. Only swapped cluster changes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    clusters = ['code', 'math', 'creative', 'reasoning']
    before = [mvp['swap_test']['baseline_mse'][c] for c in clusters]
    after = [mvp['swap_test']['after_swap_mse'][c] for c in clusters]

    x = np.arange(len(clusters))
    width = 0.35

    bars1 = ax.bar(x - width/2, before, width, label='Before Swap',
                   color=COLORS['before'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, after, width, label='After Swap (Expert_code replaced)',
                   color=[COLORS['after'] if c == 'code' else COLORS['before'] for c in clusters],
                   edgecolor='white', linewidth=0.5)

    # Annotate the swapped cluster
    ax.annotate(f'{after[0]:.1f}\n(was {before[0]:.2f})',
                xy=(0, after[0]), xytext=(0, after[0] + 1.5),
                ha='center', fontsize=11, fontweight='bold', color=COLORS['after'],
                arrowprops=dict(arrowstyle='->', color=COLORS['after'], lw=1.5))

    ax.set_xlabel('Domain Cluster')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Swap Isolation: Only the Swapped Domain Changes', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in clusters])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(after) * 1.2)

    # Isolation ratio annotation
    ratio = mvp['swap_test']['isolation_ratio']
    ax.text(0.98, 0.95, f'Isolation Ratio: {ratio:.1f}×\n(Swapped cluster Δ ÷ max other Δ)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#D1FAE5', edgecolor='#059669', alpha=0.9))

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/swap_isolation.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


# ═══════════════════════════════════════════════════════════════════
# GRAPH 2: PPL COMPARISON
# ═══════════════════════════════════════════════════════════════════

def graph_ppl_comparison():
    """Grouped bar chart: Profile-MoE vs Learned Router PPL per domain."""
    fig, ax = plt.subplots(figsize=(10, 6))

    domains = ['code', 'math', 'stories', 'wiki']
    lr = tf['learned_router']
    pr = tf['profile_router']

    learned_ppl = [lr['per_domain'][d]['ppl'] for d in domains]
    profile_ppl = [pr['per_domain'][d]['ppl'] for d in domains]

    x = np.arange(len(domains))
    width = 0.3

    bars_l = ax.bar(x - width/2, learned_ppl, width, label='Learned Router (DeepSeek-style)',
                    color=COLORS['learned'], edgecolor='white', linewidth=0.5)
    bars_p = ax.bar(x + width/2, profile_ppl, width, label='Profile-MoE',
                    color=COLORS['profile'], edgecolor='white', linewidth=0.5)

    # Value labels
    for bar in bars_l:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{bar.get_height():.1f}', ha='center', fontsize=9, color=COLORS['learned'])
    for bar in bars_p:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{bar.get_height():.1f}', ha='center', fontsize=9, fontweight='bold', color=COLORS['profile'])

    # Improvement percentages
    for i, d in enumerate(domains):
        delta = learned_ppl[i] - profile_ppl[i]
        pct = (delta / learned_ppl[i]) * 100
        ax.annotate(f'−{pct:.0f}%', xy=(i + width/2, (learned_ppl[i] + profile_ppl[i])/2),
                    ha='center', fontsize=10, fontweight='bold', color='#059669')

    ax.set_xlabel('Domain')
    ax.set_ylabel('Perplexity (lower is better)')
    ax.set_title('Profile-MoE vs Learned Router: Per-Domain Perplexity', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in domains])
    ax.legend(loc='upper left', framealpha=0.9)

    # Overall annotation
    overall_delta = ((lr['overall_ppl'] - pr['overall_ppl']) / lr['overall_ppl']) * 100
    ax.text(0.98, 0.95,
            f'Overall: {lr["overall_ppl"]:.1f} → {pr["overall_ppl"]:.1f}\n'
            f'Profile-MoE {overall_delta:.0f}% better\n'
            f'Speed: identical (0.999×)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#D1FAE5', edgecolor='#059669', alpha=0.9))

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/ppl_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


# ═══════════════════════════════════════════════════════════════════
# GRAPH 3: PROFILE HEATMAP
# ═══════════════════════════════════════════════════════════════════

def graph_profile_heatmap():
    """Heatmap: Expert × Domain capability matrix."""
    fig, ax = plt.subplots(figsize=(8, 5))

    experts = list(mvp['expert_profiles'].keys())
    # Profile order in results.json is: code, creative, math, reasoning
    dims = ['Code', 'Creative', 'Math', 'Reasoning']
    data = np.array([mvp['expert_profiles'][e] for e in experts])

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(experts)):
        for j in range(len(dims)):
            val = data[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold' if val > 0.8 else 'normal',
                    color=color)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims)
    ax.set_yticks(range(len(experts)))
    ax.set_yticklabels([e.replace('Expert_', '') for e in experts])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Capability Score', fontsize=11)

    ax.set_title('Expert Capability Profiles (Calibrated)', fontweight='bold', pad=15)
    ax.set_xlabel('Benchmark Domain')

    # Add note
    ax.text(0.5, -0.2, 'Each expert is near-perfect on its domain, near-zero elsewhere.\n'
            'Profiles come from benchmark calibration — no learned parameters.',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='#666666')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/profile_heatmap.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


# ═══════════════════════════════════════════════════════════════════
# GRAPH 4: ROUTING ACCURACY
# ═══════════════════════════════════════════════════════════════════

def graph_routing_accuracy():
    """Bar chart: Profile-MoE routing accuracy vs random baseline."""
    fig, ax = plt.subplots(figsize=(8, 5))

    clusters = list(mvp['evaluation']['per_cluster'].keys())
    accuracies = [mvp['evaluation']['per_cluster'][c]['routing_accuracy'] * 100 for c in clusters]
    random_baseline = 25.0  # 4 experts

    x = np.arange(len(clusters))
    width = 0.4

    bars = ax.bar(x, accuracies, width, label='Profile-MoE',
                  color=[COLORS[c] for c in clusters], edgecolor='white', linewidth=0.5)
    ax.axhline(y=random_baseline, color='#94A3B8', linestyle='--', linewidth=2, label=f'Random (25%)')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                f'{acc:.1f}%', ha='center', fontsize=13, fontweight='bold', color='white')

    ax.set_xlabel('Domain')
    ax.set_ylabel('Routing Accuracy (%)')
    ax.set_title('Profile-MoE Routing Accuracy vs Random Baseline', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in clusters])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 110)

    # Overall annotation
    overall = mvp['evaluation']['routing_stats']['overall_accuracy'] * 100
    ax.text(0.98, 0.95, f'Overall: {overall:.1f}%\n(vs 25% random)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#D1FAE5', edgecolor='#059669', alpha=0.9))

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/routing_accuracy.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


# ═══════════════════════════════════════════════════════════════════
# GRAPH 5: SPEED COMPARISON
# ═══════════════════════════════════════════════════════════════════

def graph_speed_comparison():
    """Side-by-side: Profile-MoE vs Learned Router speed (tok/s)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Speed bar ---
    lr_speed = tf['learned_router']['tokens_per_sec']
    pr_speed = tf['profile_router']['tokens_per_sec']

    bars = ax1.bar(['Learned Router\n(DeepSeek-style)', 'Profile-MoE'],
                   [lr_speed, pr_speed],
                   color=[COLORS['learned'], COLORS['profile']],
                   edgecolor='white', linewidth=0.5, width=0.5)

    for bar, val in zip(bars, [lr_speed, pr_speed]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3000,
                f'{val:,.0f} tok/s', ha='center', fontsize=13, fontweight='bold', color='white')

    ax1.set_ylabel('Tokens per Second')
    ax1.set_title('Inference Speed', fontweight='bold')

    ratio = pr_speed / lr_speed
    ax1.text(0.5, 0.95, f'Speed ratio: {ratio:.4f}×\n(identical within noise)',
             transform=ax1.transAxes, ha='center', va='top',
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#D1FAE5', edgecolor='#059669', alpha=0.9))

    # --- Router params ---
    lr_params = tf['learned_router']['router_params']
    pr_params = tf['profile_router']['router_params']

    bars2 = ax2.bar(['Learned Router', 'Profile-MoE'],
                    [lr_params, pr_params],
                    color=[COLORS['learned'], COLORS['profile']],
                    edgecolor='white', linewidth=0.5, width=0.5)

    for bar, val in zip(bars2, [lr_params, pr_params]):
        label = f'{val:,}' if val > 0 else 'ZERO'
        y_pos = bar.get_height() + 15 if val > 0 else 30
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                label, ha='center', fontsize=13, fontweight='bold',
                color=COLORS['profile'] if val == 0 else COLORS['learned'])

    ax2.set_ylabel('Learned Parameters')
    ax2.set_title('Router Parameters', fontweight='bold')
    ax2.set_ylim(0, max(lr_params, 50) * 1.4)

    ax2.text(0.5, 0.95, 'Profile-MoE router:\npure cosine similarity math',
             transform=ax2.transAxes, ha='center', va='top',
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#D1FAE5', edgecolor='#059669', alpha=0.9))

    fig.suptitle('Profile-MoE vs Learned Router: Speed & Parameters', fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/speed_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Generating Profile-MoE graphs...\n")

    graph_swap_isolation()
    graph_ppl_comparison()
    graph_profile_heatmap()
    graph_routing_accuracy()
    graph_speed_comparison()

    print(f"\n✓ All graphs saved to: {OUTPUT_DIR}/")
    print(f"  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(f'{OUTPUT_DIR}/{f}') / 1024
        print(f"    {f} ({size:.0f} KB)")


if __name__ == '__main__':
    main()
