"""
Builds the polished 12-slide Light-Theme Presentation for K-CARN:
- Slide 1: Title Slide (K-CARN Overview)
- Slide 2: The Core Problem: Multimodal Confounders & Benchmark Context
- Slide 3: [NEW WORKFLOW SLIDE in Light Red] Complete Baseline (RGCL) End-to-End State Flow & Workflow
- Slide 4: [REVISION SLIDE in Light Red] Baseline Limitations: The 3 Core Bottlenecks
- Slide 5: Proposed K-CARN Architecture: End-to-End State Flow
- Slide 6: Enhancement 1: Bidirectional Multi-Head Cross-Attention (Visual Grounding)
- Slide 7: Enhancement 2: Cognitive Knowledge Rationales & Adaptive Sigmoid Gate
- Slide 8: Enhancement 3: Hard-Negative Triplet Metric Learning, Hybrid BCE & Top-20 Voting
- Slide 9: Quantitative Results: Facebook Hateful Memes 4-Stage Ablation
- Slide 10: Quantitative Results: HarMeme Benchmark Generalization
- Slide 11: Engineering Pipeline, Reliability Fixes & Live Local GPU Inference
- Slide 12: Conclusion, Key Research Impact & Future Directions
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "ppt_assets")
os.makedirs(ASSET_DIR, exist_ok=True)

# -------------------------------------------------------------
# THEME PALETTE
# -------------------------------------------------------------
NAVY = RGBColor(0x14, 0x2D, 0x50)       # Deep Navy Header
AZURE = RGBColor(0x02, 0x84, 0xC7)      # Accent Blue
TEAL = RGBColor(0x0D, 0x94, 0x88)       # Positive / Gain Accent
RED = RGBColor(0xDC, 0x26, 0x26)        # Flaw / Warning Accent
BG_LIGHT = RGBColor(0xF8, 0xFA, 0xFC)   # Off-White Slide Background
CARD_BG = RGBColor(0xFF, 0xFF, 0xFF)    # Crisp White Card
CARD_BORDER = RGBColor(0xCB, 0xD5, 0xE1)
TINT_BLUE = RGBColor(0xF0, 0xF9, 0xFF)
TINT_GREEN = RGBColor(0xEC, 0xFD, 0xF5)

# Light Red Theme for Baseline Revision Slides
TINT_RED = RGBColor(0xFE, 0xF2, 0xF2)
LIGHT_RED_BG = RGBColor(0xFF, 0xF5, 0xF5)
LIGHT_RED_BORDER = RGBColor(0xFC, 0xA5, 0xA5)
DARK_RED = RGBColor(0x99, 0x1B, 0x1B)
BANNER_RED = RGBColor(0x7F, 0x1D, 0x1D)
CRIMSON = RGBColor(0xDC, 0x26, 0x26)

TEXT_DARK = RGBColor(0x1E, 0x29, 0x3B)
TEXT_MUTED = RGBColor(0x47, 0x55, 0x69)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# -------------------------------------------------------------
# PART 1: HIGH-DPI DIAGRAMS & CHARTS (300 DPI)
# -------------------------------------------------------------
def _draw_box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#142D50", tc="#1E293B",
              fontsize=9.5, bold=True, lw=1.6, subtext=None, subcolor="#475569"):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    if subtext:
        ax.text(x + w / 2, y + h * 0.63, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                color=tc, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, subtext, ha="center", va="center",
                fontsize=fontsize - 1.6, color=subcolor, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                color=tc, zorder=4)


def _arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.8, rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.28,head_length=0.45",
            color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}"
        ),
        zorder=2
    )


def build_diagrams():
    paths = {}

    # 1. Complete Baseline RGCL Workflow Diagram (Light Red Theme)
    fig, ax = plt.subplots(figsize=(12.0, 4.3), dpi=300)
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Light Red Outer Container
    outer_box = patches.FancyBboxPatch(
        (0.12, 0.12), 11.76, 4.06,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="#FFF5F5", edgecolor="#FCA5A5", linewidth=1.5, zorder=1
    )
    ax.add_patch(outer_box)

    # Step 1: Raw Inputs
    _draw_box(ax, 0.35, 2.75, 1.85, 1.0, "Raw Meme Image", fc="#FFFFFF", ec="#EF4444", tc="#991B1B",
              subtext="PNG / JPG Pixels", subcolor="#7F1D1D")
    _draw_box(ax, 0.35, 0.65, 1.85, 1.0, "Raw Meme Caption", fc="#FFFFFF", ec="#EF4444", tc="#991B1B",
              subtext="OCR Text String", subcolor="#7F1D1D")

    # Step 2: Frozen Encoders
    _draw_box(ax, 2.7, 2.75, 2.15, 1.0, "CLIP Vision Encoder", fc="#FFFFFF", ec="#DC2626", tc="#991B1B",
              subtext="ViT-L/14@336 (Frozen)\nv in R^1024", subcolor="#7F1D1D")
    _draw_box(ax, 2.7, 0.65, 2.15, 1.0, "CLIP Text Encoder", fc="#FFFFFF", ec="#DC2626", tc="#991B1B",
              subtext="Transformer (Frozen)\nt in R^768", subcolor="#7F1D1D")

    # Step 3: Blind Concatenation
    _draw_box(ax, 5.35, 1.5, 2.15, 1.4, "Naive Concat\n[v ; t]", fc="#FEE2E2", ec="#B91C1C", tc="#7F1D1D",
              subtext="Direct Vector Stacking\n[v ; t] in R^1792\n(No Cross-Attention)", subcolor="#991B1B", fontsize=10.0)

    # Step 4: Projection MLP & Memory Bank
    _draw_box(ax, 7.95, 2.2, 1.85, 1.25, "3-Layer MLP\nProjection", fc="#FFFFFF", ec="#DC2626", tc="#991B1B",
              subtext="Linear(1792->1024)\nLinear(1024->1024)\nLinear(1024->1)", subcolor="#7F1D1D", fontsize=9.0)
    _draw_box(ax, 7.95, 0.5, 1.85, 1.25, "FAISS Memory Bank\n(L2 Index)", fc="#FFFFFF", ec="#DC2626", tc="#991B1B",
              subtext="8,500 Cached Train\nEmbeddings", subcolor="#7F1D1D", fontsize=9.0)

    # Step 5: Outputs
    _draw_box(ax, 10.2, 2.2, 1.55, 1.25, "Raw BCE Logits\nClassification", fc="#FEF2F2", ec="#991B1B", tc="#7F1D1D",
              subtext="P(Hate) = sigmoid(z)\nThreshold = 0.50", subcolor="#991B1B", fontsize=8.8)
    _draw_box(ax, 10.2, 0.5, 1.55, 1.25, "Top-5 Nearest\nMean Voting", fc="#FEF2F2", ec="#991B1B", tc="#7F1D1D",
              subtext="Unweighted K-NN\nMajority Vote", subcolor="#991B1B", fontsize=8.8)

    # Arrows (Red accented)
    _arrow(ax, 2.2, 3.25, 2.68, 3.25, color="#DC2626")
    _arrow(ax, 2.2, 1.15, 2.68, 1.15, color="#DC2626")

    _arrow(ax, 4.85, 3.25, 5.33, 2.4, color="#DC2626")
    _arrow(ax, 4.85, 1.15, 5.33, 2.0, color="#DC2626")

    _arrow(ax, 7.5, 2.4, 7.93, 2.7, color="#DC2626")
    _arrow(ax, 7.5, 2.0, 7.93, 1.25, color="#DC2626")

    _arrow(ax, 9.8, 2.82, 10.18, 2.82, color="#991B1B")
    _arrow(ax, 9.8, 1.12, 10.18, 1.12, color="#991B1B")

    # Step Header Badges
    stages = [
        (1.28, "Step 1: Input Modalities"),
        (3.78, "Step 2: Frozen Encoders"),
        (6.42, "Step 3: Blind Fusion"),
        (8.88, "Step 4: Projection & Memory"),
        (10.98, "Step 5: Baseline Output")
    ]
    for x_pos, title in stages:
        ax.text(x_pos, 4.02, title, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="#991B1B", bbox=dict(boxstyle="round,pad=0.22", fc="#FEE2E2", ec="#FCA5A5", lw=1.0))

    p = os.path.join(ASSET_DIR, "diag_baseline_workflow.png")
    plt.tight_layout(pad=0.1)
    plt.savefig(p, dpi=300, facecolor="#FFFFFF")
    plt.close()
    paths["baseline_workflow"] = p

    # 2. Full K-CARN End-to-End State Flow Diagram
    fig, ax = plt.subplots(figsize=(11.8, 4.3), dpi=300)
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Inputs
    _draw_box(ax, 0.15, 3.2, 1.75, 0.85, "Meme Image", fc="#EFF6FF", ec="#0284C7", subtext="CLIP ViT-L/14 (1024d)")
    _draw_box(ax, 0.15, 1.85, 1.75, 0.85, "Meme Caption", fc="#EFF6FF", ec="#0284C7", subtext="CLIP Text (768d)")
    _draw_box(ax, 0.15, 0.4, 1.75, 0.95, "Knowledge Stream", fc="#ECFDF5", ec="#0D9488", tc="#065F46",
              subtext="Cognitive Rationale\nCLIP Text (768d)", subcolor="#0F766E")

    # Projection + Bi-Cross-Attention
    _draw_box(ax, 2.35, 2.5, 1.45, 1.55, "Modality\nProjection", fc="#F8FAFC", ec="#142D50",
              subtext="Linear -> 1024d\n+ L2 Norm")
    _draw_box(ax, 4.25, 2.25, 2.35, 1.8, "Enhancement 1:\nBidirectional\nCross-Attention",
              fc="#E0F2FE", ec="#0284C7", tc="#0C4A6E",
              subtext="8 Heads (T->V & V->T)\nResidual + LayerNorm + FFN", subcolor="#0369A1")

    # Adaptive Knowledge Gate
    _draw_box(ax, 4.25, 0.4, 2.35, 1.15, "Enhancement 2:\nAdaptive Knowledge Gate",
              fc="#ECFDF5", ec="#0D9488", tc="#065F46",
              subtext="G = sigmoid(W_g[H ; K])\nH_final = H + G * K_att", subcolor="#0F766E")

    # Joint Representation + Triplet/Hybrid Loss
    _draw_box(ax, 7.1, 1.65, 2.05, 1.55, "Enhancement 3:\nTriplet + Hard-Neg\n+ Hybrid BCE",
              fc="#FEF3C7", ec="#D97706", tc="#78350F",
              subtext="3-Layer MLP (1024d)\nMargin eps=0.1 + CE Head", subcolor="#92400E")

    # Retrieval & Output
    _draw_box(ax, 9.6, 2.45, 2.2, 1.2, "FAISS Top-20\nArithmetic Voting",
              fc="#142D50", ec="#142D50", tc="#FFFFFF",
              subtext="Similarity-Weighted\nCase Retrieval", subcolor="#CBD5E1")
    _draw_box(ax, 9.6, 0.75, 2.2, 1.15, "Final Prediction",
              fc="#DCFCE7", ec="#16A34A", tc="#14532D",
              subtext="Hateful / Benign\n+ Calibrated Prob", subcolor="#166534")

    # Arrows
    _arrow(ax, 1.92, 3.62, 2.33, 3.55)
    _arrow(ax, 1.92, 2.27, 2.33, 2.95)
    _arrow(ax, 3.82, 3.25, 4.23, 3.25, color="#0284C7", lw=2.0)
    _arrow(ax, 5.42, 2.23, 5.42, 1.58, color="#0D9488", lw=2.0)
    _arrow(ax, 1.92, 0.87, 4.23, 0.87, color="#0D9488", lw=2.0)
    _arrow(ax, 6.62, 1.15, 7.08, 2.15, color="#D97706", lw=2.0)
    _arrow(ax, 6.62, 3.1, 7.08, 2.65, color="#0284C7", lw=1.6)
    _arrow(ax, 9.17, 2.65, 9.58, 3.0, color="#142D50", lw=2.0)
    _arrow(ax, 9.17, 2.1, 9.58, 1.45, color="#16A34A", lw=2.0)
    _arrow(ax, 10.7, 2.43, 10.7, 1.93, color="#142D50", lw=1.8)

    for x_pos, lbl in [(1.0, "1. Tri-Modal Inputs"), (5.4, "2. Deep Interactive Fusion"),
                       (8.1, "3. Metric + CE Training"), (10.7, "4. Retrieval Decision")]:
        ax.text(x_pos, 4.25, lbl, ha="center", va="center", fontsize=9.0, fontweight="bold", color="#142D50")

    p = os.path.join(ASSET_DIR, "diag_kcarn_flow.png")
    plt.tight_layout(pad=0.1)
    plt.savefig(p, dpi=300, facecolor="#FFFFFF")
    plt.close()
    paths["kcarn_flow"] = p

    # 3. Bidirectional Multi-Head Cross-Attention Diagram
    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=300)
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    _draw_box(ax, 0.3, 3.5, 2.2, 0.75, "Visual Feature (v)", fc="#EFF6FF", ec="#0284C7", subtext="Projected 1024-d")
    _draw_box(ax, 3.9, 3.5, 2.2, 0.75, "Text Feature (t)", fc="#ECFDF5", ec="#0D9488", subtext="Projected 1024-d")

    _draw_box(ax, 0.2, 1.8, 2.65, 1.1, "Stream B: Image -> Text", fc="#E0F2FE", ec="#0284C7", tc="#0C4A6E",
              subtext="Q = Image,  K,V = Text\n8 Heads (d_k = 128)\nChecks: Does text twist image?")
    _draw_box(ax, 3.55, 1.8, 2.65, 1.1, "Stream A: Text -> Image", fc="#D1FAE5", ec="#0D9488", tc="#065F46",
              subtext="Q = Text,  K,V = Image\n8 Heads (d_k = 128)\nChecks: Who is text mocking?")

    _draw_box(ax, 1.0, 0.25, 4.4, 0.95, "Concat [v_att ; t_att] + Residual + LayerNorm + GELU FFN",
              fc="#142D50", ec="#142D50", tc="#FFFFFF",
              subtext="Output: Deeply Grounded 2048-d Multimodal Vector H", subcolor="#CBD5E1")

    _arrow(ax, 1.4, 3.48, 1.4, 2.93, color="#0284C7", lw=2.0)
    _arrow(ax, 5.0, 3.48, 5.0, 2.93, color="#0D9488", lw=2.0)
    _arrow(ax, 4.3, 3.48, 2.2, 2.93, color="#0D9488", lw=1.7)
    _arrow(ax, 2.1, 3.48, 4.2, 2.93, color="#0284C7", lw=1.7)
    _arrow(ax, 1.5, 1.78, 2.3, 1.23, color="#142D50", lw=1.8)
    _arrow(ax, 4.9, 1.78, 4.1, 1.23, color="#142D50", lw=1.8)

    p = os.path.join(ASSET_DIR, "diag_cross_attn.png")
    plt.tight_layout(pad=0.1)
    plt.savefig(p, dpi=300, facecolor="#FFFFFF")
    plt.close()
    paths["cross_attn"] = p

    # 4. Knowledge Rationale + Adaptive Sigmoid Gate Diagram
    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=300)
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    _draw_box(ax, 0.25, 3.55, 5.9, 0.78, "1. Cognitive Rationale Extractor (14,970 Memes)",
              fc="#ECFDF5", ec="#0D9488", tc="#065F46",
              subtext="Flags: Demographics | Historical Tropes | Dehumanization | Sarcasm")
    _draw_box(ax, 0.25, 2.35, 2.55, 0.78, "CLIP Text Encoder", fc="#F8FAFC", ec="#142D50",
              subtext="Knowledge Vector K (768-d)")
    _draw_box(ax, 3.6, 2.35, 2.55, 0.78, "Cross-Attended H", fc="#E0F2FE", ec="#0284C7",
              subtext="Image + Text Vector (2048-d)")

    _draw_box(ax, 1.15, 1.15, 4.1, 0.85, "Learned Sigmoid Gate:  G = sigmoid(W_g [H ; K_att])",
              fc="#FEF3C7", ec="#D97706", tc="#78350F",
              subtext="G -> 1 (Helpful Trope: OPEN)   |   G -> 0 (Noisy Text: CLOSE)")

    _draw_box(ax, 1.15, 0.15, 4.1, 0.7, "H_final = LayerNorm( H  +  G * K_att )",
              fc="#142D50", ec="#142D50", tc="#FFFFFF", fontsize=9.5)

    _arrow(ax, 1.5, 3.53, 1.5, 3.16, color="#0D9488", lw=1.8)
    _arrow(ax, 1.5, 2.33, 2.3, 2.03, color="#0D9488", lw=1.8)
    _arrow(ax, 4.9, 2.33, 4.1, 2.03, color="#0284C7", lw=1.8)
    _arrow(ax, 3.2, 1.13, 3.2, 0.88, color="#142D50", lw=2.0)

    p = os.path.join(ASSET_DIR, "diag_knowledge_gate.png")
    plt.tight_layout(pad=0.1)
    plt.savefig(p, dpi=300, facecolor="#FFFFFF")
    plt.close()
    paths["knowledge_gate"] = p

    # 5. Triplet Hard-Negative Embedding Space Diagram
    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=300)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    frame = patches.FancyBboxPatch((0.15, 0.15), 5.9, 4.0, boxstyle="round,pad=0.05,rounding_size=0.1",
                                   fc="#F8FAFC", ec="#CBD5E1", lw=1.5)
    ax.add_patch(frame)
    ax.text(3.1, 3.92, "Embedding Space Optimization (Triplet + Hard Negatives)",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#142D50")

    circle = patches.Circle((2.4, 2.1), 1.25, fill=False, ls="--", ec="#0284C7", lw=1.5)
    ax.add_patch(circle)
    ax.text(2.4, 0.72, "Margin Boundary (eps = 0.1)", ha="center", fontsize=8.3, color="#0284C7", fontweight="bold")

    ax.scatter([2.4], [2.1], s=320, color="#142D50", zorder=5)
    ax.text(2.4, 1.65, "Anchor Meme\n(Hateful)", ha="center", fontsize=8.3, fontweight="bold", color="#142D50")

    ax.scatter([1.45], [2.85], s=260, color="#0D9488", zorder=5)
    ax.text(1.15, 3.25, "Pseudo-Gold\nPositive (Hateful)", ha="center", fontsize=8.0, fontweight="bold", color="#065F46")

    ax.scatter([3.45], [2.55], s=240, color="#FCA5A5", edgecolors="#DC2626", lw=1.5, zorder=5)
    ax.text(3.55, 2.98, "Hard Negative\n(Look-Alike Benign)", ha="center", fontsize=8.0, fontweight="bold", color="#991B1B")

    ax.scatter([5.15], [2.55], s=260, color="#DC2626", zorder=5)
    ax.text(5.15, 2.02, "Pushed Outside\nMargin!", ha="center", fontsize=8.0, fontweight="bold", color="#991B1B")

    _arrow(ax, 1.6, 2.72, 2.22, 2.25, color="#0D9488", lw=2.2)
    ax.text(1.75, 2.35, "PULL", fontsize=8.0, fontweight="bold", color="#0D9488")

    _arrow(ax, 3.65, 2.55, 4.92, 2.55, color="#DC2626", lw=2.2)
    ax.text(4.3, 2.72, "PUSH APART", ha="center", fontsize=8.0, fontweight="bold", color="#DC2626")

    p = os.path.join(ASSET_DIR, "diag_triplet_space.png")
    plt.tight_layout(pad=0.1)
    plt.savefig(p, dpi=300, facecolor="#FFFFFF")
    plt.close()
    paths["triplet_space"] = p

    # 6. Facebook Hateful Memes 4-Stage Ablation Chart
    metrics = ["Val Accuracy", "Val AUROC", "Val Precision", "Val F1"]
    s1 = [66.40, 73.92, 71.17, 56.17]
    s2 = [66.40, 73.69, 72.29, 61.64]
    s3 = [66.00, 73.47, 74.39, 62.08]
    s4 = [71.20, 80.53, 81.63, 67.98]

    x = np.arange(len(metrics))
    w = 0.19
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    b1 = ax.bar(x - 1.5 * w, s1, w, label="1. Baseline RGCL (Concat)", color="#94A3B8")
    b2 = ax.bar(x - 0.5 * w, s2, w, label="2. + Bi-Cross-Attention", color="#38BDF8")
    b3 = ax.bar(x + 0.5 * w, s3, w, label="3. + Knowledge Gate (Naive)", color="#14B8A6")
    b4 = ax.bar(x + 1.5 * w, s4, w, label="4. Final Optimized K-CARN", color="#142D50")

    for bars, highlight in [(b1, False), (b2, False), (b3, False), (b4, True)]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.7, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.5 if not highlight else 8.2,
                    fontweight="bold" if highlight else "normal",
                    color="#142D50" if highlight else "#475569")

    ax.set_ylim(45, 91)
    ax.set_ylabel("Score (%)", fontsize=9.5, fontweight="bold", color="#142D50")
    ax.set_title("Facebook Hateful Memes (dev_seen): 4-Stage Ablation", fontsize=11, fontweight="bold", color="#142D50", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=8.0, ncol=2)

    p = os.path.join(ASSET_DIR, "chart_fb_ablation.png")
    plt.tight_layout()
    plt.savefig(p, dpi=300, facecolor="#FFFFFF")
    plt.close()
    paths["chart_fb"] = p

    # 7. HarMeme 4-Stage Ablation Chart
    metrics_h = ["Val Acc", "Val AUROC", "Test Acc", "Test AUROC", "Test F1"]
    h1 = [57.63, 59.79, 63.66, 65.74, 64.46]
    h2 = [58.76, 60.93, 61.97, 66.69, 62.40]
    h3 = [60.45, 62.93, 63.10, 65.99, 64.31]
    h4 = [60.45, 64.47, 65.35, 70.72, 67.02]

    x = np.arange(len(metrics_h))
    w = 0.19
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    b1 = ax.bar(x - 1.5 * w, h1, w, label="1. Baseline RGCL (Concat)", color="#94A3B8")
    b2 = ax.bar(x - 0.5 * w, h2, w, label="2. + Bi-Cross-Attention", color="#38BDF8")
    b3 = ax.bar(x + 0.5 * w, h3, w, label="3. + Knowledge Gate (Naive)", color="#14B8A6")
    b4 = ax.bar(x + 1.5 * w, h4, w, label="4. Final Optimized K-CARN", color="#142D50")

    for bars, highlight in [(b1, False), (b2, False), (b3, False), (b4, True)]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.55, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.0 if not highlight else 7.8,
                    fontweight="bold" if highlight else "normal",
                    color="#142D50" if highlight else "#475569")

    ax.set_ylim(50, 77)
    ax.set_ylabel("Score (%)", fontsize=9.5, fontweight="bold", color="#142D50")
    ax.set_title("HarMeme Benchmark (dev & test): 4-Stage Ablation", fontsize=11, fontweight="bold", color="#142D50", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_h, fontsize=9.0, fontweight="bold", color="#1E293B")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=8.0, ncol=2)

    p = os.path.join(ASSET_DIR, "chart_harmeme_ablation.png")
    plt.tight_layout()
    plt.savefig(p, dpi=300, facecolor="#FFFFFF")
    plt.close()
    paths["chart_harmeme"] = p

    return paths


# -------------------------------------------------------------
# PART 2: POWERPOINT HELPERS
# -------------------------------------------------------------
def send_to_back(slide, shape):
    spTree = slide.shapes._spTree
    elem = shape._element
    spTree.remove(elem)
    spTree.insert(2, elem)


def bring_to_front(slide, shape):
    spTree = slide.shapes._spTree
    elem = shape._element
    spTree.remove(elem)
    spTree.append(elem)


def set_slide_bg(slide, color=BG_LIGHT):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header_and_footer(slide, title_text, slide_num, total_slides=12, banner_color=NAVY, stripe_color=AZURE):
    set_slide_bg(slide, BG_LIGHT)

    # Top Banner
    banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.02))
    banner.fill.solid()
    banner.fill.fore_color.rgb = banner_color
    banner.line.fill.background()
    send_to_back(slide, banner)

    # Accent Stripe
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.02), Inches(13.333), Inches(0.06))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = stripe_color
    stripe.line.fill.background()

    # Standard Title Placeholder
    title = slide.shapes.title
    title.left = Inches(0.55)
    title.top = Inches(0.16)
    title.width = Inches(12.2)
    title.height = Inches(0.75)
    tf = title.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = "Calibri"
    p.font.size = Pt(25)
    p.font.bold = True
    p.font.color.rgb = WHITE
    bring_to_front(slide, title)

    # Footer Bar
    footer_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(7.08), Inches(12.333), Inches(0.015))
    footer_line.fill.solid()
    footer_line.fill.fore_color.rgb = CARD_BORDER
    footer_line.line.fill.background()

    f_left = slide.shapes.add_textbox(Inches(0.5), Inches(7.13), Inches(9.5), Inches(0.3))
    fp = f_left.text_frame.paragraphs[0]
    fp.text = "K-CARN: Knowledge-Augmented Cross-Attentive Retrieval Network for Hateful Meme Detection"
    fp.font.name = "Calibri"
    fp.font.size = Pt(10.5)
    fp.font.color.rgb = TEXT_MUTED

    f_right = slide.shapes.add_textbox(Inches(11.5), Inches(7.13), Inches(1.33), Inches(0.3))
    fr = f_right.text_frame.paragraphs[0]
    fr.alignment = PP_ALIGN.RIGHT
    fr.text = f"Slide {slide_num} / {total_slides}"
    fr.font.name = "Calibri"
    fr.font.size = Pt(10.5)
    fr.font.bold = True
    fr.font.color.rgb = banner_color


def add_card(slide, left, top, width, height, title=None, bullets=None,
             fill_color=CARD_BG, border_color=CARD_BORDER, title_color=NAVY,
             accent_bar_color=None, font_size=13.5, space_after=6, title_size=15.5):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = fill_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.0)

    if accent_bar_color:
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top + 0.06), Inches(0.08), Inches(height - 0.12))
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_bar_color
        bar.line.fill.background()

    pad_l = 0.22 if accent_bar_color else 0.18
    tb = slide.shapes.add_textbox(Inches(left + pad_l), Inches(top + 0.12), Inches(width - pad_l - 0.15), Inches(height - 0.22))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    first = True
    if title:
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Calibri"
        p.font.size = Pt(title_size)
        p.font.bold = True
        p.font.color.rgb = title_color
        p.space_after = Pt(space_after)
        first = False

    if bullets:
        for b in bullets:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.space_after = Pt(space_after)
            p.line_spacing = 1.15
            if isinstance(b, tuple):
                lead, rest = b
                r1 = p.add_run()
                r1.text = "• " + lead + " "
                r1.font.name = "Calibri"
                r1.font.size = Pt(font_size)
                r1.font.bold = True
                r1.font.color.rgb = title_color
                r2 = p.add_run()
                r2.text = rest
                r2.font.name = "Calibri"
                r2.font.size = Pt(font_size)
                r2.font.color.rgb = TEXT_DARK
            else:
                p.text = "• " + b
                p.font.name = "Calibri"
                p.font.size = Pt(font_size)
                p.font.color.rgb = TEXT_DARK
    return card


def add_fitted_image(slide, img_path, left, top, max_w, max_h, add_border=True, border_color=CARD_BORDER):
    im = Image.open(img_path)
    iw, ih = im.size
    ratio = iw / ih
    w = max_w
    h = w / ratio
    if h > max_h:
        h = max_h
        w = h * ratio
    x = left + (max_w - w) / 2
    y = top + (max_h - h) / 2

    if add_border:
        frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(max_w), Inches(max_h))
        frame.adjustments[0] = 0.02
        frame.fill.solid()
        frame.fill.fore_color.rgb = WHITE
        frame.line.color.rgb = border_color
        frame.line.width = Pt(1.0)

    slide.shapes.add_picture(img_path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))


# -------------------------------------------------------------
# PART 3: BUILD THE 12-SLIDE DECK (< 13 SLIDES)
# -------------------------------------------------------------
def build_presentation():
    imgs = build_diagrams()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    title_layout = prs.slide_layouts[0]
    content_layout = prs.slide_layouts[5]
    TOTAL = 12

    # ==================== SLIDE 1: TITLE SLIDE ====================
    s1 = prs.slides.add_slide(title_layout)
    set_slide_bg(s1, BG_LIGHT)

    card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(1.1), Inches(11.63), Inches(5.3))
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1.5)
    send_to_back(s1, card)

    lbar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.85), Inches(1.1), Inches(0.25), Inches(5.3))
    lbar.fill.solid()
    lbar.fill.fore_color.rgb = NAVY
    lbar.line.fill.background()

    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.55), Inches(1.55), Inches(4.6), Inches(0.45))
    pill.adjustments[0] = 0.5
    pill.fill.solid()
    pill.fill.fore_color.rgb = TINT_BLUE
    pill.line.color.rgb = AZURE
    pp = pill.text_frame.paragraphs[0]
    pp.text = "MULTIMODAL AI & CONTENT SAFETY RESEARCH"
    pp.alignment = PP_ALIGN.CENTER
    pp.font.name = "Calibri"
    pp.font.size = Pt(11.5)
    pp.font.bold = True
    pp.font.color.rgb = AZURE

    t = s1.shapes.title
    t.left = Inches(1.55)
    t.top = Inches(2.2)
    t.width = Inches(10.3)
    t.height = Inches(1.6)
    tp = t.text_frame.paragraphs[0]
    tp.text = "K-CARN: Enhancing Multimodal Hateful Meme Detection"
    tp.font.name = "Calibri"
    tp.font.size = Pt(36)
    tp.font.bold = True
    tp.font.color.rgb = NAVY
    bring_to_front(s1, t)

    sub = s1.placeholders[1]
    sub.left = Inches(1.55)
    sub.top = Inches(3.85)
    sub.width = Inches(10.3)
    sub.height = Inches(1.2)
    sp = sub.text_frame.paragraphs[0]
    sp.text = "From Baseline RGCL to Knowledge-Augmented Cross-Attentive Retrieval:\nBidirectional Attention, Adaptive Rationale Gating & Hard-Negative Triplet Learning"
    sp.font.name = "Calibri"
    sp.font.size = Pt(17)
    sp.font.color.rgb = TEXT_MUTED
    bring_to_front(s1, sub)

    meta = s1.shapes.add_textbox(Inches(1.55), Inches(5.35), Inches(10.3), Inches(0.65))
    mp = meta.text_frame.paragraphs[0]
    mp.text = "Benchmarks: Facebook Hateful Memes (10,000 memes) & HarMeme (7,096 memes)   |   Best Gain: +6.61% AUROC (80.53%) & +11.81% F1"
    mp.font.name = "Calibri"
    mp.font.size = Pt(12.5)
    mp.font.bold = True
    mp.font.color.rgb = TEAL

    # ==================== SLIDE 2: THE PROBLEM (WHY MEMES FOOL AI) ====================
    s2 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s2, "1. The Core Problem: Why Hateful Memes Fool Standard AI", 2, TOTAL)

    add_card(
        s2, 0.55, 1.32, 6.0, 2.75,
        title="The 'Hidden Meaning Trap' (Multimodal Confounders)",
        bullets=[
            ("Why simple word filters fail:", "Memes rarely use obvious slurs. Both the picture alone and text alone look 100% innocent in isolation."),
            ("Real-world example:", "Photo of a desert / garbage dump (Benign) + Caption 'Look at where they belong' (Benign) -> Combined targeting an ethnic minority = Hate Speech."),
            ("Core Challenge:", "Hate speech exists ONLY in the implicit interaction between visual signals, caption words, and cultural context.")
        ],
        accent_bar_color=AZURE, font_size=13.5
    )

    add_card(
        s2, 6.78, 1.32, 6.0, 2.75,
        title="Our Starting Point: The Baseline RGCL Project",
        bullets=[
            ("Base Framework:", "RGCL (Retrieval-Guided Contrastive Learning) built on frozen CLIP ViT-L/14-336 vision and text encoders."),
            ("Core Philosophy:", "Instead of memorizing hate rules in weights, retrieve the most similar known memes from a FAISS memory bank and vote."),
            ("Why Retrieval Matters:", "Handles fast-evolving internet slang and allows instant knowledge updates without retraining foundation models.")
        ],
        accent_bar_color=NAVY, font_size=13.5
    )

    add_card(
        s2, 0.55, 4.30, 6.0, 2.55,
        title="Benchmark 1: Facebook Hateful Memes (FB)",
        bullets=[
            ("Dataset Size:", "10,000 memes (8,500 Train | 500 Dev Seen | 1,000 Test Seen | 1,000 Test Unseen)."),
            ("Adversarial Design:", "Built with 'confounder' pairs—modifying either image or text flips the label between Hateful and Benign."),
            ("Evaluation Split:", "dev_seen (500 balanced labeled memes; official test_seen omits ground-truth labels).")
        ],
        fill_color=TINT_BLUE, border_color=AZURE, font_size=13
    )

    add_card(
        s2, 6.78, 4.30, 6.0, 2.55,
        title="Benchmark 2: HarMeme (COVID-19 & US Politics)",
        bullets=[
            ("Dataset Size:", "7,096 real-world memes (2,938 Train | 177 Dev | 355 Test labeled splits)."),
            ("Real Internet Mockery:", "Captures messy internet sarcasm, political hostility, and harmful stereotypes targeting public figures & groups."),
            ("Evaluation Value:", "Proves our enhancements generalize to uncontrolled social media environments beyond curated benchmarks.")
        ],
        fill_color=TINT_GREEN, border_color=TEAL, font_size=13
    )

    # ==================== SLIDE 3: [NEW] BASELINE COMPLETE WORKFLOW (LIGHT RED) ====================
    s3 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s3, "2. Baseline Model (RGCL): Complete End-to-End Workflow & Execution Flow", 3, TOTAL,
                          banner_color=BANNER_RED, stripe_color=CRIMSON)

    add_fitted_image(s3, imgs["baseline_workflow"], 0.55, 1.25, 12.23, 3.45, add_border=True, border_color=LIGHT_RED_BORDER)

    # 4 Quick Revision Cards at the bottom in Light Red theme
    c_w = 2.92
    c_gap = 0.18
    add_card(
        s3, 0.55, 4.88, c_w, 2.05,
        title="Step 1: Dual Inputs",
        bullets=[
            ("Vision:", "Frozen CLIP ViT-L/14@336 extracts 1024-d image embedding v."),
            ("Text:", "Frozen CLIP text transformer extracts 768-d caption embedding t.")
        ],
        fill_color=TINT_RED, border_color=LIGHT_RED_BORDER, title_color=DARK_RED, font_size=11.5, title_size=13.5
    )

    add_card(
        s3, 0.55 + (c_w + c_gap), 4.88, c_w, 2.05,
        title="Step 2: Naive Concat",
        bullets=[
            ("Direct Stacking:", "Vectors glued as [v ; t] in R^1792."),
            ("Core Limitation:", "Zero cross-attention; image pixels never interact with caption words.")
        ],
        fill_color=TINT_RED, border_color=LIGHT_RED_BORDER, title_color=DARK_RED, font_size=11.5, title_size=13.5
    )

    add_card(
        s3, 0.55 + 2 * (c_w + c_gap), 4.88, c_w, 2.05,
        title="Step 3: Linear Projection",
        bullets=[
            ("3-Layer MLP:", "Maps 1792-d -> 1024-d projection space with dropout."),
            ("Supervision:", "Trained with naive cosine similarity / basic BCE loss.")
        ],
        fill_color=TINT_RED, border_color=LIGHT_RED_BORDER, title_color=DARK_RED, font_size=11.5, title_size=13.5
    )

    add_card(
        s3, 0.55 + 3 * (c_w + c_gap), 4.88, c_w, 2.05,
        title="Step 4: Top-5 Retrieval",
        bullets=[
            ("FAISS Index:", "Queries memory bank of 8,500 training meme embeddings."),
            ("Voting:", "Simple unweighted mean vote across Top-5 nearest neighbors.")
        ],
        fill_color=TINT_RED, border_color=LIGHT_RED_BORDER, title_color=DARK_RED, font_size=11.5, title_size=13.5
    )

    # ==================== SLIDE 4: BASELINE LIMITATIONS & BOTTLENECKS (LIGHT RED) ====================
    s4 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s4, "3. Baseline Limitations: The 3 Core Technical Bottlenecks (Why It Failed)", 4, TOTAL,
                          banner_color=BANNER_RED, stripe_color=CRIMSON)

    add_card(
        s4, 0.55, 1.35, 3.9, 5.5,
        title="Flaw 1: Blind Feature Gluing (Linear Addition Trap)",
        bullets=[
            ("The Mechanism:", "Baseline simply concatenates [v in R^1024 ; t in R^768] into a single 1792-d vector."),
            ("Mathematical Failure:", "When fed into the first linear layer, W[v ; t] = W_v v + W_t t. The vision features and text features are merely ADDED independently!"),
            ("Why It Fails on Memes:", "Sarcasm and irony depend entirely on cross-multiplicative interaction (e.g. caption 'bravery' contrasting with cowardice in the image). Linear addition cannot compute cross-modal correlation."),
            ("Consequence:", "Model misclassifies benign confounders that share common words or common image objects.")
        ],
        fill_color=TINT_RED, border_color=LIGHT_RED_BORDER, title_color=DARK_RED, font_size=12.2, space_after=8
    )

    add_card(
        s4, 4.71, 1.35, 3.9, 5.5,
        title="Flaw 2: Zero Cultural Context & Historical Dogwhistles",
        bullets=[
            ("The Mechanism:", "Relies 100% on literal zero-shot embeddings from frozen CLIP."),
            ("Semantic Blindspot:", "CLIP's pre-training recognizes literal entities ('desert', 'well-dressed man', 'swimming pool') but possesses no reasoning regarding historical tropes, racial stereotypes, or disguised slurs."),
            ("Real Example Missed:", "Meme displaying a photo of people walking with caption 'it is time to send these parasites back'. CLIP sees innocent people and terrain; it cannot deduce the dehumanizing immigrant trope without background knowledge."),
            ("Consequence:", "High false-negative rate on subtle, coded hate speech.")
        ],
        fill_color=TINT_RED, border_color=LIGHT_RED_BORDER, title_color=DARK_RED, font_size=12.2, space_after=8
    )

    add_card(
        s4, 8.88, 1.35, 3.9, 5.5,
        title="Flaw 3: Brittle Loss Function & Narrow Top-5 Voting",
        bullets=[
            ("The Mechanism:", "Trained with naive cosine similarity without hard-negative mining; static Top-5 unweighted mean voting during inference."),
            ("Optimization Collapse:", "Because training samples are dominated by easy pairs, the model quickly overfits by Epoch 2 and stops learning fine-grained boundaries."),
            ("Retrieval Fragility:", "Taking an unweighted mean of only 5 neighbors is heavily polluted if even 1 or 2 retrieved cases belong to a confusing confounder."),
            ("Consequence:", "FB dev AUROC stalled at 73.92% and F1 at 56.17%, with test recall collapsing on subtle political memes.")
        ],
        fill_color=TINT_RED, border_color=LIGHT_RED_BORDER, title_color=DARK_RED, font_size=12.2, space_after=8
    )

    # ==================== SLIDE 5: PROPOSED K-CARN ARCHITECTURE ====================
    s5 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s5, "4. Our Enhanced System: K-CARN End-to-End State Flow", 5, TOTAL)

    add_fitted_image(s5, imgs["kcarn_flow"], 0.55, 1.26, 12.23, 4.05, add_border=True)

    add_card(
        s5, 0.55, 5.48, 12.23, 1.42,
        title="Three Modular Upgrades Built on Top of RGCL (100% Backward-Compatible)",
        bullets=[
            ("1. Bidirectional Cross-Attention:", "Replaces naive concatenation with 8-head Text<->Image attention so words ground directly onto visual cues."),
            ("2. Gated Knowledge Rationales:", "Injects a 3rd semantic stream (14,970 meme rationales) filtered through a learned Sigmoid Gate G in [0,1]."),
            ("3. Optimized Metric + CE Training:", "Triplet Margin Loss + Hard-Negative Mining + Hybrid BCE + Top-20 Similarity-Weighted Arithmetic Voting.")
        ],
        fill_color=TINT_BLUE, border_color=AZURE, font_size=12.5, space_after=3
    )

    # ==================== SLIDE 6: ENHANCEMENT 1 - CROSS ATTENTION ====================
    s6 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s6, "5. Enhancement 1: Bidirectional Multi-Head Cross-Attention", 6, TOTAL)

    add_fitted_image(s6, imgs["cross_attn"], 0.55, 1.32, 6.2, 5.5, add_border=True)

    add_card(
        s6, 6.95, 1.32, 5.83, 2.65,
        title="Simple Intuition: Two Teams of 8 Detectives",
        bullets=[
            ("Old Way (Concat):", "Taping a photo and a note together without either side reading the other."),
            ("New Way (Bidirectional):", "Text queries Image ('Who is being mocked?') AND Image queries Text ('Does this caption twist my harmless photo?')."),
            ("Real FB Example:", "'white people is this a shooting range' -> links 'shooting range' straight to the crowded visual scene.")
        ],
        accent_bar_color=AZURE, font_size=13
    )

    add_card(
        s6, 6.95, 4.15, 5.83, 2.67,
        title="Technical Implementation (cross_attention_fusion.py)",
        bullets=[
            ("Scaled Dot-Product:", "Attention(Q,K,V) = Softmax(Q * K^T / sqrt(d_k)) * V across 8 parallel heads (d_k = 128)."),
            ("Stability Layers:", "L2-normalized 1024-d projections + Residual Add + LayerNorm + GELU Feed-Forward refinement."),
            ("Measured Advantage:", "Boosted FB Val F1 from 0.562 -> 0.616 (+5.5%) and HarMeme Val Acc from 57.6% -> 58.8%.")
        ],
        fill_color=TINT_BLUE, border_color=AZURE, font_size=13
    )

    # ==================== SLIDE 7: ENHANCEMENT 2 - KNOWLEDGE GATE ====================
    s7 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s7, "6. Enhancement 2: Cognitive Knowledge Rationales & Adaptive Gate", 7, TOTAL)

    add_fitted_image(s7, imgs["knowledge_gate"], 0.55, 1.32, 6.2, 5.5, add_border=True)

    add_card(
        s7, 6.95, 1.32, 5.83, 2.65,
        title="Why Add a 3rd Knowledge Stream?",
        bullets=[
            ("The Gap:", "CLIP sees objects and words, but misses coded slurs, historical tropes, and mocking sarcasm."),
            ("Offline Extraction:", "generate_rationales.py analyzed all 14,970 memes (FB + HarMeme) and encoded them into 768-d CLIP vectors."),
            ("Real FB Example:", "Caption 'bravery at its finest' -> Rationale flags: 'No explicit slur; employs mockery_sarcasm juxtaposed with image.'")
        ],
        accent_bar_color=TEAL, font_size=13
    )

    add_card(
        s7, 6.95, 4.15, 5.83, 2.67,
        title="Why an Adaptive Sigmoid Gate? (Hallucination Guard)",
        bullets=[
            ("Formula:", "G = sigmoid(W_g[H ; K_att] + b_g),   H_final = LayerNorm(H + G * K_att)."),
            ("Smart Filtering:", "Opens (G -> 1) when rationale clarifies a hidden trope; closes (G -> 0) when text is generic or noisy."),
            ("Measured Advantage:", "Eliminated late-epoch overfitting: HarMeme Val Acc jumped to 60.45% (AUROC 0.629) & FB Precision to 74.39%.")
        ],
        fill_color=TINT_GREEN, border_color=TEAL, font_size=13
    )

    # ==================== SLIDE 8: ENHANCEMENT 3 - TRIPLET + HYBRID ====================
    s8 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s8, "7. Enhancement 3: Hard-Negative Triplet Loss, Hybrid CE & Top-20", 8, TOTAL)

    add_fitted_image(s8, imgs["triplet_space"], 0.55, 1.32, 6.1, 5.5, add_border=True)

    add_card(
        s8, 6.85, 1.32, 5.93, 2.7,
        title="Learning From the Trickiest Look-Alikes",
        bullets=[
            ("Hard Negative Mining:", "Instead of easy random comparisons, FAISS finds the closest opposite-class meme (a benign look-alike) and forces a margin eps = 0.1."),
            ("Pseudo-Gold Positives:", "Pulls each meme toward its closest same-class neighbor to form tight semantic clusters."),
            ("Why It Works:", "Directly solves Multimodal Confounders where two nearly identical memes have opposite labels.")
        ],
        accent_bar_color=NAVY, font_size=13
    )

    add_card(
        s8, 6.85, 4.2, 5.93, 2.62,
        title="Hybrid Supervision + Top-20 Arithmetic Voting",
        bullets=[
            ("Hybrid Loss:", "L_total = L_Triplet_HardNeg + lambda * L_BCE jointly trains the retrieval space AND calibrates the classifier head."),
            ("Top-20 Similarity Voting:", "Replaced unweighted Top-5 mean with Top-20 arithmetic similarity-weighted voting (batch_size=64, 20 epochs)."),
            ("Measured Advantage:", "Pushed FB AUROC to 80.53% (+6.61%) and HarMeme Test AUROC to 70.72% (+4.98%).")
        ],
        fill_color=TINT_BLUE, border_color=AZURE, font_size=13
    )

    # ==================== SLIDE 9: FB RESULTS & ABLATION ====================
    s9 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s9, "8. Results: Facebook Hateful Memes (4-Stage Ablation)", 9, TOTAL)

    add_fitted_image(s9, imgs["chart_fb"], 0.55, 1.32, 7.7, 5.5, add_border=True)

    add_card(
        s9, 8.45, 1.32, 4.33, 2.65,
        title="Net Gains Over Baseline (dev_seen)",
        bullets=[
            ("Val AUROC:", "73.92% -> 80.53%  (+6.61%)"),
            ("Val Accuracy:", "66.40% -> 71.20%  (+4.80%)"),
            ("Val Precision:", "71.17% -> 81.63%  (+10.46%)"),
            ("Val F1 Score:", "56.17% -> 67.98%  (+11.81%)")
        ],
        fill_color=TINT_GREEN, border_color=TEAL, title_color=TEAL, font_size=13.5, space_after=8
    )

    add_card(
        s9, 8.45, 4.15, 4.33, 2.67,
        title="What the Ablation Proves",
        bullets=[
            ("Cross-Attention:", "Immediately boosted F1 by +5.47% over blind concat."),
            ("Knowledge Gate:", "Lifted precision to 74.39% with fewer false alarms."),
            ("Full Optimized K-CARN:", "Broke 80.5% AUROC & 81.6% Precision—high reliability for real moderation.")
        ],
        accent_bar_color=NAVY, font_size=13
    )

    # ==================== SLIDE 10: HARMEME RESULTS & ABLATION ====================
    s10 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s10, "9. Results: HarMeme Benchmark (Cross-Dataset Generalization)", 10, TOTAL)

    add_fitted_image(s10, imgs["chart_harmeme"], 0.55, 1.32, 7.7, 5.5, add_border=True)

    add_card(
        s10, 8.45, 1.32, 4.33, 2.75,
        title="Net Gains Over Baseline (dev & test)",
        bullets=[
            ("Test AUROC:", "65.74% -> 70.72%  (+4.98%)"),
            ("Val AUROC:", "59.79% -> 64.47%  (+4.68%)"),
            ("Val Accuracy:", "57.63% -> 60.45%  (+2.82%)"),
            ("Test F1 Score:", "64.46% -> 67.02%  (+2.56%)"),
            ("Test Accuracy:", "63.66% -> 65.35%  (+1.69%)")
        ],
        fill_color=TINT_GREEN, border_color=TEAL, title_color=TEAL, font_size=13, space_after=5
    )

    add_card(
        s10, 8.45, 4.25, 4.33, 2.57,
        title="Key Generalization Insights",
        bullets=[
            ("Monotonic Val Gain:", "Every stage climbed steadily (57.6% -> 58.8% -> 60.5%)."),
            ("Anti-Overfitting:", "Knowledge gating prevented Epoch-4 decay seen in baseline."),
            ("High Recall:", "Optimized K-CARN reached 77.2% test recall on harmful political memes.")
        ],
        accent_bar_color=AZURE, font_size=12.8
    )

    # ==================== SLIDE 11: ENGINEERING & LOCAL INFERENCE ====================
    s11 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s11, "10. Engineering Pipeline, Reliability Fixes & Local GPU Demo", 11, TOTAL)

    add_card(
        s11, 0.55, 1.32, 6.0, 2.7,
        title="Permanent Pipeline Fixes Engineered in src/",
        bullets=[
            ("Auto-Path Resolution:", "dataset.py automatically maps HuggingFace '/' model names to '_HF.pt' checkpoints (no manual symlinks)."),
            ("FAISS & PyTorch 2.x Fixes:", "Fixed numpy/Tensor detach crashes in retrieval.py & run_rac.py, and PyTorch weights_only=False loading."),
            ("FB Numeric ID Safety:", "Fixed EasyDict TypeError by normalizing integer batch IDs to strings in rac_dataloader.py & evaluate_rac.py.")
        ],
        accent_bar_color=NAVY, font_size=13
    )

    add_card(
        s11, 6.78, 1.32, 6.0, 2.7,
        title="Modular Design & Reproducibility",
        bullets=[
            ("Zero Breaking Changes:", "--fusion_mode concat runs exact baseline; --fusion_mode k_carn --use_knowledge True runs full K-CARN."),
            ("Saved Embeddings:", "Pre-extracted CLIP + Knowledge embeddings allow 20-epoch Kaggle GPU runs in ~2-9 minutes."),
            ("Interactive Web Dashboard:", "Built full-featured Streamlit GUI (app.py) with real-time Auto-OCR and live model predictions.")
        ],
        accent_bar_color=AZURE, font_size=13
    )

    add_card(
        s11, 0.55, 4.22, 12.23, 2.6,
        title="Live Local GPU Inference Pipeline (predict.py on Local NVIDIA CUDA)",
        bullets=[
            ("Single-Command Prediction:", "conda run -n RGCL python predict.py --image 'data/image/FB/All/08291.png' --text 'white people is this a shooting range'"),
            ("Verified Live Output:", "Sample #1 (08291.png): Ground Truth = HATEFUL | Predicted = HATEFUL  [CORRECT]"),
            ("Verified Live Output:", "Sample #2 (46971.png, 'bravery at its finest'): Ground Truth = HATEFUL | Predicted = HATEFUL  [CORRECT]"),
            ("Verified Live Output:", "Sample #3 (03745.png, 'white privilege discount...'): Ground Truth = HATEFUL | Predicted = HATEFUL  [CORRECT]")
        ],
        fill_color=TINT_BLUE, border_color=AZURE, font_size=13, space_after=5
    )

    # ==================== SLIDE 12: CONCLUSION & FUTURE WORK ====================
    s12 = prs.slides.add_slide(content_layout)
    add_header_and_footer(s12, "11. Conclusion, Key Contributions & Future Directions", 12, TOTAL)

    add_card(
        s12, 0.55, 1.32, 6.0, 2.6,
        title="1. Research Impact",
        bullets=[
            ("Solved Blind Fusion:", "Replaced naive concatenation with Bidirectional Multi-Head Cross-Attention so visual and text signals actively ground each other."),
            ("Solved Missing Context:", "Added a Gated Cognitive Knowledge stream across 14,970 memes to surface hidden tropes without hallucination.")
        ],
        accent_bar_color=NAVY, font_size=13.5
    )

    add_card(
        s12, 6.78, 1.32, 6.0, 2.6,
        title="2. Proven Quantitative Gains",
        bullets=[
            ("Facebook Hateful Memes:", "AUROC 73.92% -> 80.53% (+6.61%), Accuracy +4.80%, Precision +10.46%, F1 +11.81%."),
            ("HarMeme Cross-Dataset:", "Test AUROC 65.74% -> 70.72% (+4.98%), Val AUROC +4.68%, Test F1 +2.56%.")
        ],
        fill_color=TINT_GREEN, border_color=TEAL, title_color=TEAL, font_size=13.5
    )

    add_card(
        s12, 0.55, 4.12, 6.0, 2.7,
        title="3. Complete Research Deliverables",
        bullets=[
            ("Enhanced Codebase:", "Modular K-CARN PyTorch modules + local GPU inference script (predict.py) + Streamlit GUI (app.py)."),
            ("Documentation:", "3-page technical report (K_CARN_Project_Report.pdf) + GitHub repo (ESHANT05/rm_project).")
        ],
        accent_bar_color=AZURE, font_size=13.5
    )

    add_card(
        s12, 6.78, 4.12, 6.0, 2.7,
        title="4. Future Directions",
        bullets=[
            ("End-to-End LoRA Fine-Tuning:", "Unfreeze top CLIP vision/text layers with low-rank adapters (r=8) for domain-specific representations."),
            ("Hybrid BM25 + Visual Heatmaps:", "Combine dense FAISS with sparse BM25 keyword retrieval and overlay cross-attention maps on memes for moderator explainability.")
        ],
        accent_bar_color=TEAL, font_size=13.5
    )

    out_path = os.path.join(BASE_DIR, "RGCL_Paper_Presentation.pptx")
    backup_path = os.path.join(BASE_DIR, "RGCL_Paper_Presentation_Updated.pptx")
    prs.save(backup_path)
    print(f"Successfully saved updated deck to {backup_path}")
    try:
        prs.save(out_path)
        print(f"Successfully generated {out_path} with {len(prs.slides)} slides!")
    except PermissionError:
        print(f"Note: {out_path} is currently open in PowerPoint. Changes saved to {backup_path}")


if __name__ == "__main__":
    build_presentation()
