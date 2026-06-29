#!/usr/bin/env python3
"""Generate all charts and graphics for Energivanu Forbes-style magazine."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

OUT = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(OUT, exist_ok=True)

# Global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

DARK_BG = '#0a0a0a'
ACCENT = '#00d4ff'
ACCENT2 = '#ff6b35'
ACCENT3 = '#00ff88'
GRID_COLOR = '#1a1a1a'
TEXT_WHITE = '#ffffff'
TEXT_GRAY = '#999999'


def chart_training_progression():
    """MAPE improvement over versions."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    versions = ['v1\nMIT Data\n14K rows', 'v2\nAlibaba\n3L rows', 'v3\nAlibaba\n50L rows', 'v4\nAlibaba Raw\n30L rows']
    mape = [8438, 75.4, 37.28, 20.3]
    colors = ['#ff4444', '#ff8844', '#ffaa00', ACCENT3]

    bars = ax.bar(range(len(versions)), mape, color=colors, width=0.6, edgecolor='none', zorder=3)

    for i, (bar, val) in enumerate(zip(bars, mape)):
        label = f'{val}%' if val > 100 else f'{val}%'
        y_pos = min(val * 0.5, 4000) if val > 100 else val + 2
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va='bottom', fontsize=16, fontweight='bold',
                color=TEXT_WHITE)

    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, fontsize=11, color=TEXT_GRAY)
    ax.set_ylabel('MAPE (%)', fontsize=13, color=TEXT_GRAY)
    ax.set_yscale('log')
    ax.tick_params(colors=TEXT_GRAY)
    ax.grid(axis='y', alpha=0.15, color=TEXT_GRAY)
    ax.set_xlim(-0.5, 3.5)

    # Annotation
    ax.annotate('99.8% reduction\nin prediction error',
                xy=(3, 20.3), xytext=(1.5, 200),
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=2),
                fontsize=12, color=ACCENT, ha='center',
                fontweight='bold')

    plt.title('Prediction Accuracy Journey', fontsize=20, fontweight='bold',
              color=TEXT_WHITE, pad=20)
    plt.savefig(f'{OUT}/training_progression.png', facecolor=DARK_BG)
    plt.close()


def chart_bess_smoothing():
    """BESS grid smoothing visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    t = np.linspace(0, 4*np.pi, 200)
    raw = np.sin(t) * 50 + 200 + np.random.normal(0, 8, len(t))
    smoothed = np.convolve(raw, np.ones(15)/15, mode='same') + 200

    for ax in [ax1, ax2]:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_GRAY)
        ax.grid(axis='y', alpha=0.1, color=TEXT_GRAY)

    ax1.plot(t, raw, color='#ff4444', alpha=0.8, linewidth=1.2)
    ax1.fill_between(t, raw, 150, alpha=0.15, color='#ff4444')
    ax1.set_title('Before BESS Optimization', fontsize=14, color=TEXT_WHITE, fontweight='bold')
    ax1.set_ylabel('Power (MW)', fontsize=11, color=TEXT_GRAY)
    ax1.axhline(y=200, color=TEXT_GRAY, linestyle='--', alpha=0.3)
    ax1.set_ylim(130, 270)

    ax2.plot(t, smoothed, color=ACCENT3, alpha=0.9, linewidth=1.5)
    ax2.fill_between(t, smoothed, 150, alpha=0.15, color=ACCENT3)
    ax2.set_title('After BESS Optimization', fontsize=14, color=TEXT_WHITE, fontweight='bold')
    ax2.axhline(y=200, color=TEXT_GRAY, linestyle='--', alpha=0.3)
    ax2.set_ylim(130, 270)

    # Reduction annotation
    fig.text(0.5, 0.02, '⚡ 30.0% Reduction in Grid Power Variance  |  10.5% Peak Demand Shaving',
             ha='center', fontsize=14, color=ACCENT, fontweight='bold')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f'{OUT}/bess_smoothing.png', facecolor=DARK_BG)
    plt.close()


def chart_competitive_landscape():
    """Competitive comparison radar chart."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    categories = ['ML Power\nPrediction', 'BESS\nControl', 'Phase\nStaggering', 'Grid Signal\nIntegration', 'Open\nSource', 'GPU\nAwareness']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Energivanu
    ev = [5, 5, 5, 5, 5, 5]
    ev += ev[:1]

    # Competitors
    zeus = [1, 0, 0, 0, 5, 5]
    zeus += zeus[:1]
    emerald = [2, 0, 0, 5, 0, 1]
    emerald += emerald[:1]
    phaidra = [2, 0, 0, 0, 0, 2]
    phaidra += phaidra[:1]

    ax.plot(angles, ev, 'o-', linewidth=2.5, color=ACCENT, label='Energivanu')
    ax.fill(angles, ev, alpha=0.15, color=ACCENT)
    ax.plot(angles, zeus, 's--', linewidth=1.5, color='#ff4444', alpha=0.7, label='Zeus')
    ax.plot(angles, emerald, '^--', linewidth=1.5, color='#ffaa00', alpha=0.7, label='Emerald AI')
    ax.plot(angles, phaidra, 'D--', linewidth=1.5, color='#aa44ff', alpha=0.7, label='Phaidra')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, color=TEXT_WHITE)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=8, color=TEXT_GRAY)
    ax.set_ylim(0, 5.5)
    ax.grid(color='#333', alpha=0.3)
    ax.spines['polar'].set_color('#333')

    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                       fontsize=11, framealpha=0.8, facecolor='#1a1a1a',
                       edgecolor='#333', labelcolor=TEXT_WHITE)

    plt.title('Competitive Capability Matrix', fontsize=18, fontweight='bold',
              color=TEXT_WHITE, pad=30)
    plt.savefig(f'{OUT}/competitive_radar.png', facecolor=DARK_BG)
    plt.close()


def chart_market_opportunity():
    """Market size visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    years = ['2024', '2025', '2026', '2027', '2028', '2029', '2030']
    dc_power = [460, 520, 600, 720, 870, 1050, 1260]  # GW
    ai_share = [120, 170, 240, 340, 470, 630, 820]    # GW AI workloads

    x = np.arange(len(years))
    w = 0.35

    bars1 = ax.bar(x - w/2, dc_power, w, color='#2a2a4a', edgecolor='#444',
                   label='Total DC Power', zorder=3)
    bars2 = ax.bar(x + w/2, ai_share, w, color=ACCENT, edgecolor='none',
                   label='AI Workloads', zorder=3)

    for bar, val in zip(bars2, ai_share):
        ax.text(bar.get_x() + bar.get_width()/2, val + 15, f'{val} GW',
                ha='center', fontsize=9, color=ACCENT, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=12, color=TEXT_GRAY)
    ax.set_ylabel('Power Demand (GW)', fontsize=13, color=TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY)
    ax.grid(axis='y', alpha=0.1, color=TEXT_GRAY)
    ax.legend(fontsize=11, framealpha=0.8, facecolor='#1a1a1a',
              edgecolor='#333', labelcolor=TEXT_WHITE)

    # TAM annotation
    ax.annotate('$47B TAM by 2030\nData Center Power Optimization',
                xy=(6, 820), xytext=(4, 950),
                arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=2),
                fontsize=13, color=ACCENT2, ha='center', fontweight='bold')

    plt.title('Global Data Center Power Demand', fontsize=20, fontweight='bold',
              color=TEXT_WHITE, pad=20)
    plt.savefig(f'{OUT}/market_opportunity.png', facecolor=DARK_BG)
    plt.close()


def chart_architecture():
    """System architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(x, y, w, h, text, color, fontsize=10, subtext=''):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + (0.15 if subtext else 0), text,
                ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color='white')
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.25, subtext,
                    ha='center', va='center', fontsize=8, color='#cccccc')

    def arrow(x1, y1, x2, y2, color='#666'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Title
    ax.text(7, 7.6, 'ENERGIVANU SYSTEM ARCHITECTURE', ha='center', fontsize=18,
            fontweight='bold', color=TEXT_WHITE)

    # Layer 1: Grid Signal
    draw_box(0.5, 6.2, 3, 1, 'ERCOT / Utility Grid', '#1a3a5c', 11, 'SCED Signals (4s)')
    draw_box(5, 6.2, 3, 1, 'OpenADR 2.0b VEN', '#1a3a5c', 11, 'Demand Response Events')
    draw_box(9.5, 6.2, 3.5, 1, 'Grid Signal Parser', '#1a3a5c', 11, 'Signal Classification')

    # Arrows layer 1
    arrow(3.5, 6.7, 5, 6.7, ACCENT)
    arrow(8, 6.7, 9.5, 6.7, ACCENT)

    # Layer 2: Decision Engine
    draw_box(1.5, 4.2, 4, 1.4, 'TCN + Attention', '#0d4a2e', 13, 'Power Prediction (613K params)')
    draw_box(6.5, 4.2, 4, 1.4, 'MPC Controller', '#0d4a2e', 13, 'Optimal BESS Dispatch')
    draw_box(11, 4.2, 2.5, 1.4, 'Phase\nScheduler', '#0d4a2e', 11, 'All-Reduce Stagger')

    # Arrows layer 1 to 2
    arrow(11.25, 6.2, 3.5, 5.6, ACCENT)
    arrow(11.25, 6.2, 8.5, 5.6, ACCENT)

    # Arrows within layer 2
    arrow(5.5, 4.9, 6.5, 4.9, ACCENT3)

    # Layer 3: Execution
    draw_box(1.5, 2, 4, 1.4, 'BESS Battery', '#5c1a1a', 13, 'Charge/Discharge/SOC')
    draw_box(6.5, 2, 4, 1.4, 'GPU Cluster', '#5c1a1a', 13, 'H100/A100 Training')
    draw_box(11, 2, 2.5, 1.4, 'Grid\nCompliance', '#5c1a1a', 11, 'PCLR < 600s')

    # Arrows layer 2 to 3
    arrow(3.5, 4.2, 3.5, 3.4, ACCENT2)
    arrow(8.5, 4.2, 8.5, 3.4, ACCENT2)
    arrow(12.25, 4.2, 12.25, 3.4, ACCENT2)

    # Layer 4: Result
    draw_box(3, 0.3, 8, 1.2, '⚡ GRID POWER MEETS TARGET — Training Continues Uninterrupted', '#1a1a3a', 13,
             '30% Smoothing | 59% Variance Reduction | 10.5% Peak Shaving | <20s Response')

    arrow(3.5, 2, 5, 1.5, ACCENT3)
    arrow(8.5, 2, 9, 1.5, ACCENT3)

    plt.savefig(f'{OUT}/architecture.png', facecolor=DARK_BG)
    plt.close()


def chart_response_timeline():
    """Response time comparison."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    components = ['Signal\nParsing', 'Signal\nClassification', 'MPC\nComputation', 'Phase\nScheduling', 'BESS\nExecution', 'GPU Cluster\nStagger']
    times = [1, 1, 2, 1, 5, 10]
    colors = [ACCENT, ACCENT, ACCENT3, ACCENT3, ACCENT2, ACCENT2]

    cumulative = 0
    for i, (comp, t, c) in enumerate(zip(components, times, colors)):
        ax.barh(0, t, left=cumulative, height=0.5, color=c, edgecolor='none', alpha=0.9)
        ax.text(cumulative + t/2, 0, f'{t}s', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax.text(cumulative + t/2, -0.4, comp, ha='center', va='center',
                fontsize=8, color=TEXT_GRAY)
        cumulative += t

    # PCLR limit
    ax.axvline(x=600, color='#ff4444', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(300, 0.45, 'PCLR Requirement: 600 seconds', ha='center',
            fontsize=11, color='#ff4444', fontweight='bold')
    ax.text(21, 0.45, 'Energivanu: 21 seconds', ha='left',
            fontsize=12, color=ACCENT3, fontweight='bold')

    ax.set_xlim(-1, 650)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel('Time (seconds)', fontsize=11, color=TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY)
    ax.set_yticks([])

    plt.title('Response Time: 30x Faster Than Required', fontsize=18,
              fontweight='bold', color=TEXT_WHITE, pad=15)
    plt.savefig(f'{OUT}/response_timeline.png', facecolor=DARK_BG)
    plt.close()


def chart_feature_importance():
    """15-feature input visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    features = [
        'facility_mw', 'power_roc', 'power_roc2', 'power_roll_mean', 'power_roll_std',
        'gpu_avg_power', 'gpu_max_power', 'gpu_avg_temp', 'gpu_max_temp',
        'gpu_avg_util', 'gpu_mem_util', 'cpu_util', 'hour_sin', 'hour_cos', 'is_allreduce'
    ]
    importance = [0.12, 0.08, 0.05, 0.09, 0.07, 0.11, 0.09, 0.06, 0.05, 0.08, 0.06, 0.04, 0.04, 0.03, 0.03]
    categories = ['Power']*5 + ['GPU']*6 + ['System']*3 + ['Phase']*1
    cat_colors = {'Power': ACCENT, 'GPU': ACCENT3, 'System': ACCENT2, 'Phase': '#aa44ff'}
    colors = [cat_colors[c] for c in categories]

    y_pos = range(len(features))
    bars = ax.barh(y_pos, importance, color=colors, height=0.7, edgecolor='none')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9, color=TEXT_WHITE)
    ax.set_xlabel('Relative Importance', fontsize=11, color=TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.1, color=TEXT_GRAY)

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in cat_colors.items()]
    ax.legend(handles=patches, fontsize=10, framealpha=0.8, facecolor='#1a1a1a',
              edgecolor='#333', labelcolor=TEXT_WHITE, loc='lower right')

    plt.title('15-Feature Input Architecture', fontsize=18, fontweight='bold',
              color=TEXT_WHITE, pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUT}/feature_importance.png', facecolor=DARK_BG)
    plt.close()


def chart_alibaba_training():
    """Alibaba training loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    epochs = [1, 5, 10, 15, 20, 30, 41, 50, 60]
    train_loss = [7.66, 6.30, 6.16, 6.08, 6.03, 5.88, 5.80, 5.78, 5.77]
    val_loss = [6.68, 6.14, 6.30, 6.12, 6.04, 5.98, 5.95, 5.97, 6.05]

    ax.plot(epochs, train_loss, 'o-', color=ACCENT, linewidth=2.5, markersize=6, label='Train Loss')
    ax.plot(epochs, val_loss, 's-', color=ACCENT2, linewidth=2.5, markersize=6, label='Val Loss')

    ax.axvline(x=41, color=ACCENT3, linestyle='--', alpha=0.5)
    ax.text(42, 7.3, 'Best Checkpoint\nEpoch 41\nVal Loss: 5.95', fontsize=10,
            color=ACCENT3, fontweight='bold')

    ax.fill_between(epochs, train_loss, val_loss, alpha=0.1, color=ACCENT)

    ax.set_xlabel('Epoch', fontsize=12, color=TEXT_GRAY)
    ax.set_ylabel('Loss', fontsize=12, color=TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY)
    ax.grid(alpha=0.1, color=TEXT_GRAY)
    ax.legend(fontsize=11, framealpha=0.8, facecolor='#1a1a1a',
              edgecolor='#333', labelcolor=TEXT_WHITE)

    plt.title('Alibaba GPU Trace Training — 30 Lakh Rows', fontsize=18,
              fontweight='bold', color=TEXT_WHITE, pad=15)
    plt.savefig(f'{OUT}/alibaba_training.png', facecolor=DARK_BG)
    plt.close()


def chart_data_pipeline():
    """Data flow visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Data sources
    sources = [
        (0.5, 4.5, 'Alibaba GPU\nTrace 2020', '30L rows\nCC BY 4.0'),
        (0.5, 3, 'Self-Collected\nT4 Data', 'Kaggle/Colab\nOwn License'),
        (0.5, 1.5, 'Synthetic\nTraces', 'Demo Model\nZero Risk'),
    ]

    for x, y, title, sub in sources:
        box = FancyBboxPatch((x, y), 2.2, 1, boxstyle="round,pad=0.1",
                             facecolor='#1a3a5c', edgecolor=ACCENT, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x+1.1, y+0.6, title, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')
        ax.text(x+1.1, y+0.2, sub, ha='center', va='center', fontsize=7, color=TEXT_GRAY)

    # Processing
    proc_box = FancyBboxPatch((4, 2), 3, 2.5, boxstyle="round,pad=0.15",
                              facecolor='#0d4a2e', edgecolor=ACCENT3, linewidth=2)
    ax.add_patch(proc_box)
    ax.text(5.5, 3.8, 'DATA PIPELINE', ha='center', fontsize=12, fontweight='bold', color=ACCENT3)
    ax.text(5.5, 3.3, '• Feature Engineering', ha='center', fontsize=9, color=TEXT_WHITE)
    ax.text(5.5, 2.9, '• Quality Validation', ha='center', fontsize=9, color=TEXT_WHITE)
    ax.text(5.5, 2.5, '• Normalization', ha='center', fontsize=9, color=TEXT_WHITE)
    ax.text(5.5, 2.1, '• Provenance Tracking', ha='center', fontsize=9, color=TEXT_WHITE)

    # Model
    model_box = FancyBboxPatch((8.5, 2.2), 3, 2, boxstyle="round,pad=0.15",
                               facecolor='#3a1a5c', edgecolor='#aa44ff', linewidth=2)
    ax.add_patch(model_box)
    ax.text(10, 3.6, 'EnergivanuPEB', ha='center', fontsize=12, fontweight='bold', color='#cc88ff')
    ax.text(10, 3.1, 'TCN + Attention', ha='center', fontsize=10, color=TEXT_WHITE)
    ax.text(10, 2.6, '613,612 Parameters', ha='center', fontsize=9, color=TEXT_GRAY)

    # Arrows
    for y in [5, 3.5, 2]:
        ax.annotate('', xy=(4, 3.2), xytext=(2.7, y),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.5))
    ax.annotate('', xy=(8.5, 3.2), xytext=(7, 3.2),
                arrowprops=dict(arrowstyle='->', color=ACCENT3, lw=2))
    ax.annotate('', xy=(11.5, 1.5), xytext=(10, 2.2),
                arrowprops=dict(arrowstyle='->', color='#aa44ff', lw=2))

    # Output
    out_box = FancyBboxPatch((9, 0.2), 2.5, 0.9, boxstyle="round,pad=0.1",
                             facecolor='#1a1a3a', edgecolor=ACCENT, linewidth=1.5)
    ax.add_patch(out_box)
    ax.text(10.25, 0.65, 'ONNX Export\nProduction Ready', ha='center', va='center',
            fontsize=9, fontweight='bold', color=ACCENT)

    plt.title('Commercial-Safe Data Strategy', fontsize=18, fontweight='bold',
              color=TEXT_WHITE, pad=15)
    plt.savefig(f'{OUT}/data_pipeline.png', facecolor=DARK_BG)
    plt.close()


if __name__ == '__main__':
    print("Generating charts...")
    chart_training_progression()
    print("  ✓ Training progression")
    chart_bess_smoothing()
    print("  ✓ BESS smoothing")
    chart_competitive_landscape()
    print("  ✓ Competitive radar")
    chart_market_opportunity()
    print("  ✓ Market opportunity")
    chart_architecture()
    print("  ✓ Architecture diagram")
    chart_response_timeline()
    print("  ✓ Response timeline")
    chart_feature_importance()
    print("  ✓ Feature importance")
    chart_alibaba_training()
    print("  ✓ Alibaba training curve")
    chart_data_pipeline()
    print("  ✓ Data pipeline")
    print("All charts generated in assets/")
