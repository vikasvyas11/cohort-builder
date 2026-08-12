# =============================================================================
# modules/report_gen.py
# PURPOSE: PDF Report following the exact prescribed structure:
#   1. Dataset information (both datasets) + completeness chart
#   2. Blocking rules + cumulative comparison count chart
#   3. Comparison methods
#   4. Model training
#       a. Match weights chart
#       b. Parameter estimates chart
#   5. Unlinkable records chart
#   6. Edge metrics + match weight histogram
#   7. Cluster metrics table + dataset overlap Venn diagram
#   8. Confusion matrix (TP/FP/FN + Precision/Recall/F1)
# =============================================================================

import io
import math
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from fpdf import FPDF

# ── Colours ─────────────────────────────────────────────────────────────────
NAVY   = (20, 40, 90)
BLUE   = (0, 90, 180)
SLATE  = (60, 70, 80)
LIGHT  = (235, 238, 245)
WHITE  = (255, 255, 255)
GREEN  = (16, 140, 80)
RED    = (192, 40, 40)
ORANGE = (210, 100, 20)

# ── Matplotlib palette ───────────────────────────────────────────────────────
C_BLUE   = "#005AB4"
C_ORANGE = "#E06820"
C_GREEN  = "#28A060"
C_GREY   = "#6B7280"
C_RED    = "#C02828"
PALETTE  = [C_BLUE, C_ORANGE, C_GREEN, "#9333EA", C_RED, "#0891B2", "#D97706"]

PAGE_W  = 210
PAGE_H  = 297
MARGIN  = 18
CONTENT = PAGE_W - 2 * MARGIN


# =============================================================================
# ── CHART HELPERS ─────────────────────────────────────────────────────────────
# =============================================================================

def _ct(text: str) -> str:
    """Sanitise text for fpdf2's latin-1 Helvetica font."""
    return (text.replace("\u2014", "-").replace("\u2013", "-")
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .encode("latin-1", errors="replace").decode("latin-1"))


def _png(fig) -> bytes:
    """Render matplotlib figure to PNG bytes, then close figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Chart 1: Column completeness ─────────────────────────────────────────────
def chart_completeness(miss_a: dict, miss_b: Optional[dict] = None) -> bytes:
    """Grouped bar chart of field completeness (%) per dataset.
    Mirrors 'Column completeness by source dataset' from the Linkage report."""
    fields = list(miss_a.keys())
    x      = np.arange(len(fields))
    has_b  = bool(miss_b)
    w      = 0.35 if has_b else 0.55

    fig, ax = plt.subplots(figsize=(9, 3.4))
    bars_a = ax.bar(x - (w/2 if has_b else 0),
                    [miss_a.get(f, 0) for f in fields],
                    width=w, label="Dataset A", color=C_BLUE, edgecolor="white", lw=0.4)
    for b in bars_a:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.8,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=7)

    if has_b:
        bars_b = ax.bar(x + w/2, [miss_b.get(f, 0) for f in fields],
                        width=w, label="Dataset B", color=C_ORANGE, edgecolor="white", lw=0.4)
        for b in bars_b:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.8,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(fields, fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Completeness (%)", fontsize=9)
    ax.set_title("Column Completeness by Source Dataset", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, lw=0.3, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if has_b:
        ax.legend(fontsize=8)
    plt.tight_layout()
    return _png(fig)


# ── Chart 2: Cumulative blocking rule comparison count ───────────────────────
def chart_cumulative_blocking(blocking_counts: list) -> bytes:
    """Horizontal stacked bar showing cumulative comparisons per blocking rule.
    Mirrors 'Cumulative count of blocking rules' from the Linkage report."""
    if not blocking_counts:
        return b""

    labels = []
    for r in blocking_counts:
        sql = r["rule_sql"]
        sql = (sql[:52] + "...") if len(sql) > 55 else sql
        labels.append(f"Rule {r['rule_index']}: {sql}")
    values = [r["n"] for r in blocking_counts]
    cumsum = np.cumsum(values)
    y      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, max(3.0, len(labels) * 0.55)))
    for i, (val, cum, color) in enumerate(zip(values, cumsum, PALETTE * 3)):
        # Stacked: plot cumulative bar (full width) then individual contribution
        ax.barh(y[i], cum, height=0.6, color=color, alpha=0.25, edgecolor="none")
        ax.barh(y[i], val, height=0.6, color=color, edgecolor="white", lw=0.4,
                label=f"Rule {i}")
        ax.text(cum + max(cumsum) * 0.01, y[i],
                f"{val:,} (+{val:,}  cumulative: {cum:,})",
                va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Comparisons Generated", fontsize=9)
    ax.set_title("Cumulative Count of Blocking Rules\n"
                 "Additional comparisons added by each rule",
                 fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _png(fig)


# ── Chart 3: Match weights ────────────────────────────────────────────────────
def chart_match_weights(model_params: dict) -> bytes:
    """Horizontal bar chart of log2(m/u) per comparison level.
    Mirrors 'Match Weights Chart' from the SeRP Trained Model section."""
    comparisons = model_params.get("comparisons", [])
    prior       = model_params.get("prior_log_odds")

    items = []
    if prior is not None:
        items.append(("Prior (starting weight)", prior, C_GREY))
    for i, comp in enumerate(comparisons):
        c = PALETTE[i % len(PALETTE)]
        for level in comp["levels"]:
            w = level.get("match_weight")
            if w is not None:
                lbl = f"{comp['field']}: {level['label']}"
                items.append(((lbl[:55] + "...") if len(lbl) > 58 else lbl, w, c))

    if not items:
        return b""

    labels, weights, colors = zip(*items)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(labels) * 0.40)))
    bars = ax.barh(y, weights, color=colors, edgecolor="white", lw=0.3, height=0.65)
    ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    for bar, w in zip(bars, weights):
        offset = 0.1 if w >= 0 else -0.1
        ax.text(w + offset, bar.get_y() + bar.get_height()/2,
                f"{w:.2f}", va="center", ha="left" if w >= 0 else "right", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Comparison Level Match Weight = log2(m/u)", fontsize=9)
    ax.set_title("Match Weights Chart\n"
                 "Model parameters (components of final match weight)",
                 fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=c["field"])
               for i, c in enumerate(comparisons)]
    if prior is not None:
        handles.insert(0, mpatches.Patch(color=C_GREY, label="Prior"))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    plt.tight_layout()
    return _png(fig)


# ── Chart 4: Parameter estimates ─────────────────────────────────────────────
def chart_parameter_estimates(model_params: dict) -> bytes:
    """Horizontal bar chart of m-probabilities as log odds per level.
    Mirrors 'Parameter Estimates Chart' from the SeRP Trained Model section."""
    comparisons = model_params.get("comparisons", [])
    items = []
    for i, comp in enumerate(comparisons):
        c = PALETTE[i % len(PALETTE)]
        for level in comp["levels"]:
            m = level.get("m_prob")
            if m and 0 < m < 1 and not level.get("is_null"):
                lo  = math.log2(m / (1.0 - m))
                lbl = f"{comp['field']}: {level['label']}"
                items.append(((lbl[:55] + "...") if len(lbl) > 58 else lbl, lo, c))
    if not items:
        return b""

    labels, logodds, colors = zip(*items)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(labels) * 0.40)))
    bars = ax.barh(y, logodds, color=colors, edgecolor="white", lw=0.3, height=0.65)
    ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    for bar, v in zip(bars, logodds):
        offset = 0.05 if v >= 0 else -0.05
        ax.text(v + offset, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("m-probability (as log odds) = log2(m / (1-m))", fontsize=9)
    ax.set_title("Parameter Estimates Chart\n"
                 "Comparison of parameter estimates across training sessions",
                 fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _png(fig)


# ── Chart 5: Unlinkable records ───────────────────────────────────────────────
def chart_unlinkables(unlinkables: dict) -> bytes:
    """Line chart of % unlinkable records vs match-weight threshold.
    Mirrors 'Unlinkable records' chart from the Linkage report."""
    thresholds = unlinkables.get("thresholds", [])
    pcts       = unlinkables.get("pcts", [])
    if not thresholds:
        return b""

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(thresholds, pcts, color=C_BLUE, lw=1.8)
    ax.fill_between(thresholds, pcts, alpha=0.12, color=C_BLUE)
    ax.set_xlabel("Threshold match weight", fontsize=9)
    ax.set_ylabel("Percentage of unlinkable records (%)", fontsize=9)
    ax.set_title("Unlinkable Records\n"
                 "Records with insufficient information to exceed a given match threshold",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(min(thresholds), max(thresholds))
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, lw=0.3, alpha=0.6)
    ax.xaxis.grid(True, lw=0.3, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _png(fig)


# ── Chart 6: Match weight histogram ──────────────────────────────────────────
def chart_match_weight_histogram(weight_dist: pd.DataFrame) -> bytes:
    """Bar chart histogram of match weights across all predicted edges.
    Mirrors 'Match Weight Histogram' from the SeRP Edge Metrics section."""
    if weight_dist.empty:
        return b""

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(weight_dist["weight_bin"], weight_dist["n_edges"],
           width=0.45, color=C_BLUE, edgecolor="white", lw=0.3)
    ax.set_xlabel("Match weight", fontsize=9)
    ax.set_ylabel("Count of record comparisons in bin", fontsize=9)
    ax.set_title("Histogram of Match Weights", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, lw=0.3, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _png(fig)


# ── Chart 7: Cluster size distribution ───────────────────────────────────────
def chart_cluster_sizes(cluster_sizes: pd.DataFrame) -> bytes:
    if cluster_sizes.empty:
        return b""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(cluster_sizes["n_nodes"].astype(str), cluster_sizes["n_clusters"],
           color=C_BLUE, edgecolor="white", lw=0.3)
    ax.set_xlabel("Cluster size (records)", fontsize=9)
    ax.set_ylabel("Number of clusters", fontsize=9)
    ax.set_title("Cluster Size Distribution", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, lw=0.3, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _png(fig)


# ── Chart 8: Dataset overlap Venn diagram ────────────────────────────────────
def draw_venn_diagram(
    a_only: int, both_ab: int, b_only: int, *,
    figsize=(6, 4.0), title_fontsize=10, number_fontsize=20, label_fontsize=9,
    color_a=C_BLUE, color_b=C_ORANGE, color_both=C_GREY,
    title="Venn Diagram of Cluster_id Set Membership\nfakea / fakeb",
):
    """Shared two-circle Venn diagram (matplotlib). Draws the figure only —
    callers decide what to do with it: embed via st.pyplot() for the in-app
    views, or convert to PNG bytes (see chart_venn_diagram below) for the
    PDF report. Consolidates what used to be three separate, near-identical
    copies of this exact drawing code (this module, p_analysis.py,
    p_compare_export.py) into one, parameterised for each context's
    existing size/color/label choices so all three keep their current
    appearance exactly as before.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    for cx, color in [(3.5, color_a), (6.5, color_b)]:
        ax.add_patch(plt.Circle((cx, 3.5), 2.5, color=color, alpha=0.35, zorder=2))
        ax.add_patch(plt.Circle((cx, 3.5), 2.5, fill=False, edgecolor=color, lw=1.5, zorder=3))

    for tx, val in [(2.4, a_only), (5.0, both_ab), (7.6, b_only)]:
        ax.text(tx, 3.5, f"{val:,}", ha="center", va="center", fontsize=number_fontsize,
                fontweight="bold", color="white", zorder=4)

    for lx, label, color in [(2.5, "Dataset A\n(A only)", color_a),
                              (5.0, "Both",               color_both),
                              (7.5, "Dataset B\n(B only)", color_b)]:
        ax.text(lx, 0.7, label, ha="center", va="center",
                fontsize=label_fontsize, fontweight="bold", color=color)

    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", y=0.98)
    fig.tight_layout()
    return fig


def chart_venn_diagram(venn: dict, operation_mode: str) -> bytes:
    """Two-circle Venn diagram showing cluster membership by source dataset,
    as PNG bytes for PDF embedding. Only rendered in link mode (two datasets)."""
    if operation_mode == "dedupe":
        return b""
    fig = draw_venn_diagram(
        venn.get("a_only", 0), venn.get("both_ab", 0), venn.get("b_only", 0),
    )
    return _png(fig)


# ── Chart 9: Confusion matrix heatmap ────────────────────────────────────────
def chart_confusion_matrix(cm: dict) -> bytes:
    """2x2-style confusion matrix heatmap (without TN which is too large).
    Shows TP, FP, FN with colour coding: green=good, red=bad."""
    tp = cm.get("tp", 0)
    fp = cm.get("fp", 0)
    fn = cm.get("fn", 0)

    # Build a 2x2 matrix array; bottom-right (TN) shown as N/A
    data = np.array([[tp, fp], [fn, float("nan")]], dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.axis("off")

    # Colour the cells manually
    cell_colors = [
        [C_GREEN,  "#B85050"],   # TP=green, FP=red
        ["#B85050", "#CCCCCC"],  # FN=red,   TN=grey
    ]
    labels = [
        [f"TP\n{tp:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", "TN\n(omitted)"],
    ]

    for row in range(2):
        for col in range(2):
            rect = plt.Rectangle([col, 1 - row], 1, 1,
                                  facecolor=cell_colors[row][col],
                                  edgecolor="white", lw=2, zorder=1)
            ax.add_patch(rect)
            ax.text(col + 0.5, 1.5 - row, labels[row][col],
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color="white", zorder=2)

    # Axis labels
    ax.text(0.5, 2.20, "Predicted Match",     ha="center", fontsize=10, fontweight="bold")
    ax.text(1.5, 2.20, "Predicted Non-Match", ha="center", fontsize=10, fontweight="bold")
    ax.text(-0.35, 1.5, "True\nMatch",       ha="center", va="center", fontsize=10, fontweight="bold", rotation=90)
    ax.text(-0.35, 0.5, "True\nNon-Match",   ha="center", va="center", fontsize=10, fontweight="bold", rotation=90)

    ax.set_xlim(-0.6, 2.1)
    ax.set_ylim(-0.1, 2.5)

    ax.set_title("Confusion Matrix (Pairwise)\n"
                 "Ground truth: 'cluster' column in original dataset",
                 fontsize=10, fontweight="bold")

    # Metrics table below
    p   = cm.get("precision", 0)
    r   = cm.get("recall", 0)
    f1  = cm.get("f1", 0)
    fs  = cm.get("fstar", 0)
    fdr = cm.get("fdr", 0)
    fnr = cm.get("fnr", 0)
    metrics_text = (
        f"Precision: {p:.4f}   Recall: {r:.4f}   F1: {f1:.4f}\n"
        f"F*: {fs:.4f}   FDR: {fdr:.4f}   FNR: {fnr:.4f}"
    )
    ax.text(0.75, -0.08, metrics_text, ha="center", va="top",
            fontsize=9, color="#3C4650", style="italic",
            transform=ax.transData)

    plt.tight_layout()
    return _png(fig)


# ── Chart 10: Precision-Recall curve ─────────────────────────────────────────
def chart_precision_recall(truth_space_df: pd.DataFrame) -> bytes:
    """Precision-Recall curve from the truth space table (example 16)."""
    if truth_space_df.empty:
        return b""
    df = truth_space_df.dropna(subset=["precision_val", "recall_val"])
    if df.empty:
        return b""

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # Left: Precision-Recall curve
    ax = axes[0]
    ax.plot(df["recall_val"], df["precision_val"], color=C_BLUE, lw=1.8)
    ax.set_xlabel("Recall (1 - FNR)", fontsize=9)
    ax.set_ylabel("Precision (1 - FDR)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall Curve", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, lw=0.3, alpha=0.6)
    ax.xaxis.grid(True, lw=0.3, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: F* vs match probability (F-star is the harmonic-like composite metric)
    ax2 = axes[1]
    df2 = truth_space_df.dropna(subset=["fstar", "match_probability"])
    if not df2.empty:
        ax2.plot(df2["match_probability"], df2["fstar"], color=C_GREEN, lw=1.8)
        ax2.set_xlabel("Match Probability Threshold", fontsize=9)
        ax2.set_ylabel("F* Score (TP / (TP + FP + FN))", fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.05)
        ax2.set_title("F* Score vs Threshold", fontsize=10, fontweight="bold")
        ax2.yaxis.grid(True, lw=0.3, alpha=0.6)
        ax2.xaxis.grid(True, lw=0.3, alpha=0.4)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    return _png(fig)


# =============================================================================
# ── PDF CLASS ─────────────────────────────────────────────────────────────────
# =============================================================================

class _LinkageReport(FPDF):
    """fpdf2 subclass with layout helpers."""

    def __init__(self, run_label: str = "Run 1"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.run_label = run_label
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.add_page()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*SLATE)
        self.cell(0, 5, "Linkage Report  |  Cohort Builder", align="L")
        self.ln(1)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.25)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*SLATE)
        self.cell(0, 5, f"{self.run_label}  |  Page {self.page_no()}", align="C")

    def h1(self, text: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*NAVY)
        self.cell(0, 8, _ct(text), ln=True)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.5)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(4)
        self.set_text_color(*SLATE)

    def h2(self, text: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*NAVY)
        self.cell(0, 6, _ct(text), ln=True)
        self.ln(1)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*SLATE)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*SLATE)
        self.multi_cell(CONTENT, 5.5, _ct(text))
        self.ln(2)

    def kv(self, label: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*NAVY)
        self.cell(62, 6, _ct(label) + ":", ln=False)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*SLATE)
        self.cell(0, 6, _ct(str(value)), ln=True)

    def img(self, png_bytes: bytes, caption: str = "", w_frac: float = 0.93):
        if not png_bytes:
            return
        w = CONTENT * w_frac
        self.image(io.BytesIO(png_bytes), x=(PAGE_W - w)/2, w=w)
        self.ln(1)
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 110)
            self.cell(0, 5, _ct(caption), align="C", ln=True)
        self.ln(2)

    def table(self, headers: list, rows: list, col_widths: Optional[list] = None):
        if col_widths is None:
            cw = CONTENT / max(len(headers), 1)
            col_widths = [cw] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, _ct(str(h)), border=0, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*SLATE)
        for i, row in enumerate(rows):
            fill = (i % 2 == 0)
            self.set_fill_color(*LIGHT) if fill else self.set_fill_color(*WHITE)
            for cell, w in zip(row, col_widths):
                self.cell(w, 6, _ct(str(cell)), border=0, fill=fill, align="C")
            self.ln()
        self.ln(3)

    def sql_block(self, sql: str):
        self.set_fill_color(240, 243, 250)
        self.set_font("Courier", "", 8)
        self.set_text_color(*NAVY)
        for line in sql.split("\n"):
            line = line.strip()
            if line:
                self.cell(0, 5, line[:120], fill=True, ln=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*SLATE)
        self.ln(1)


# =============================================================================
# ── SECTION BUILDERS ──────────────────────────────────────────────────────────
# =============================================================================

def _cover(pdf, run_config, run_label):
    """Cover page."""
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*NAVY)
    pdf.ln(22)
    pdf.cell(0, 14, "Linkage Report", align="C", ln=True)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 9, "Splink Cohort Builder", align="C", ln=True)
    pdf.ln(8)
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.8)
    pdf.line(MARGIN + 15, pdf.get_y(), PAGE_W - MARGIN - 15, pdf.get_y())
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*SLATE)
    op = run_config.get("operation_mode", "-").replace("_", " ").title()
    lt = run_config.get("linkage_type", "-").title()
    for line in [
        f"Linkage Project Name: splink_cohort_builder",
        f"Linkage Run Name: {run_label}",
        f"Linkage Type: {op}",
        f"Model Type: {lt}",
        f"Document Creation Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]:
        pdf.cell(0, 7, line, align="C", ln=True)

    pdf.ln(10)
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.3)
    pdf.line(MARGIN + 15, pdf.get_y(), PAGE_W - MARGIN - 15, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 7, "Run Configuration", ln=True)
    pdf.ln(2)
    active_br = ", ".join([f for f, v in run_config.get("blocking_toggles", {}).items() if v])
    for label, value in [
        ("Operation mode",    op),
        ("Linkage type",      lt),
        ("Fields used",       ", ".join(run_config.get("selected_fields", []))),
        ("Active blocking",   active_br),
        ("Cluster threshold", str(run_config.get("cluster_threshold", 0.8))),
    ]:
        pdf.kv(label, value)


def _datasets_section(pdf, n_input, operation_mode, fields, miss_a, miss_b):
    """Section 1: Dataset information + completeness chart."""
    pdf.add_page()
    pdf.h1("Datasets")
    pdf.body(
        "This section shows an overview of the datasets used, including linkage "
        "fields, field mapping, and column-level completeness (missingness)."
    )

    # Dataset A
    pdf.h2("Base table name: Dataset A")
    pdf.kv("Dataset", "Linkage input dataset")
    pdf.kv("Number of rows", str(n_input))
    pdf.h2("Linkage fields")
    pdf.body(", ".join(fields))
    pdf.h2("Field mapping (Dataset A)")
    pdf.table(
        ["Linkage System Alias", "Source Fieldname"],
        [[f, f] for f in fields],
        [CONTENT * 0.5, CONTENT * 0.5],
    )
    if miss_a:
        pdf.h2("Dataset statistics: Dataset A")
        pdf.kv("Missingness", "Completeness per column shown below.")
        pdf.table(
            ["Field", "Completeness (%)"],
            [[f, f"{v:.1f}%"] for f, v in miss_a.items()],
            [CONTENT * 0.6, CONTENT * 0.4],
        )

    # Dataset B (link mode only)
    if operation_mode != "dedupe" and miss_b:
        pdf.add_page()
        pdf.h2("Base table name: Dataset B (fake1000b)")
        pdf.body(
            "Dataset B is a 50% sample of Dataset A with controlled data quality "
            "errors: 14% first-name typos, 9% surname typos, 5% missing DOBs, "
            "15% email variations, 11% city abbreviations, 7% gender errors."
        )
        pdf.kv("Number of rows", str(n_input - (n_input if operation_mode == "dedupe" else 0)))
        pdf.h2("Field mapping (Dataset B)")
        pdf.table(
            ["Linkage System Alias", "Source Fieldname"],
            [[f, f] for f in fields],
            [CONTENT * 0.5, CONTENT * 0.5],
        )
        pdf.h2("Dataset statistics: Dataset B")
        pdf.table(
            ["Field", "Completeness (%)"],
            [[f, f"{v:.1f}%"] for f, v in miss_b.items()],
            [CONTENT * 0.6, CONTENT * 0.4],
        )

    # Completeness chart
    miss_b_for_chart = miss_b if operation_mode != "dedupe" else None
    if miss_a:
        pdf.add_page()
        pdf.h2("Column Completeness Chart")
        pdf.body("Completeness = percentage of non-null values per column.")
        pdf.img(chart_completeness(miss_a, miss_b_for_chart),
                "Figure: Column completeness by source dataset.")


def _blocking_section(pdf, run_config, blocking_counts):
    """Section 2: Blocking rules + cumulative comparison count chart."""
    pdf.add_page()
    pdf.h1("Blocking Rules")
    pdf.body(
        "Blocking rules generate pairwise record comparisons. Only record pairs "
        "that agree on at least one blocking field are compared. Blocking reduces "
        "the O(n^2) comparison space but may miss true matches if rules are too "
        "restrictive."
    )

    toggles    = run_config.get("blocking_toggles", {})
    count_map  = {r["rule_sql"]: r["n"] for r in blocking_counts}
    active     = [(i, f, f'l."{f}" = r."{f}"')
                  for i, (f, v) in enumerate(toggles.items()) if v]

    for idx, field, sql in active:
        pdf.h2(f"Blocking rule {idx}:")
        pdf.sql_block(sql)
        n = count_map.get(sql, "N/A")
        pdf.kv("Number of generated comparisons", f"{n:,}" if isinstance(n, int) else str(n))
        pdf.body("Reasoning: Pairs that agree on this field are brought forward for "
                 "detailed comparison. Useful when this field has high completeness "
                 "and low error rate.")

    # Cumulative blocking chart
    if blocking_counts:
        pdf.add_page()
        pdf.h2("Cumulative count of blocking rules chart")
        pdf.img(chart_cumulative_blocking(blocking_counts),
                "Figure: Additional comparisons generated by each successive blocking rule.")


def _comparison_methods_section(pdf, settings_used):
    """Section 3: Comparison methods used to score the relationship between records."""
    pdf.add_page()
    pdf.h1("Comparison Methods")
    pdf.body(
        "Comparison methods are used to score the relationship between two records. "
        "Each method defines ordered levels from exact match to total disagreement. "
        "A null level handles missing values with no evidence contribution."
    )
    comparisons = settings_used.get("comparisons", [])
    pdf.kv("Number of comparison methods for edge scoring", str(len(comparisons)))
    pdf.ln(2)

    for i, comp in enumerate(comparisons):
        field = comp.get("output_column_name", f"Field {i}")
        pdf.h2(f"Comparison {i}: {field}")
        pdf.body("SQL comparison levels (first match wins):")
        for level in comp.get("comparison_levels", []):
            sql  = level.get("sql_condition", "")
            null = level.get("is_null_level", False)
            tag  = "  [NULL LEVEL]" if null else ""
            if sql and sql.upper() != "ELSE":
                pdf.sql_block(f"{sql}{tag}")
            else:
                pdf.body("ELSE (all other comparisons)")
        pdf.body("Description: CustomComparison\nReasoning: Placeholder text.")


def _model_training_section(pdf, run_config, fields):
    """Section 4a: Model training parameters."""
    pdf.add_page()
    pdf.h1("Model Training")
    pdf.body(
        "This section shows the model training parameters and hyperparameters "
        "used for the probabilistic Fellegi-Sunter model."
    )
    PRIORITY  = ["first_name", "surname", "dob", "city", "email", "gender", "postcode"]
    available = [f for f in PRIORITY if f in fields]
    primary   = available[0] if available else (fields[0] if fields else "first_name")
    secondary = available[1] if len(available) > 1 else None

    pdf.h2("Overview")
    pdf.kv("Number of training schemes", "2 (prior estimate + EM)")
    pdf.h2("Prior Estimation")
    pdf.kv("Number of random samples", "1.00E+05")
    pdf.kv("Recall estimate", "0.6")
    pdf.body("Rule 0 (prior estimation):")
    pdf.sql_block(f'l."{primary}" = r."{primary}"')

    pdf.h2("Estimation Schemes (Expectation-Maximisation)")
    pdf.body("fix_u_probabilities = True: u-probabilities held fixed during EM.")
    pdf.h2("Training Rule 0:")
    pdf.sql_block(f'l."{primary}" = r."{primary}"')
    if secondary:
        pdf.h2("Training Rule 1:")
        pdf.sql_block(f'l."{secondary}" = r."{secondary}"')


def _trained_model_section(pdf, model_params):
    """Section 4b: Match weights chart + parameter estimates chart."""
    pdf.add_page()
    pdf.h1("Trained Model Parameters")
    pdf.body(
        "This section shows the match weights (log2(m/u)) and parameter estimates "
        "obtained from EM training. Higher positive weights = stronger match evidence."
    )

    if not model_params.get("training_complete"):
        pdf.body("Parameter extraction not available for this run (deterministic mode).")
        return

    prior = model_params.get("prior_log_odds")
    if prior is not None:
        pdf.kv("Prior (starting) match weight", f"{prior:.4f}")

    rows = []
    for comp in model_params.get("comparisons", []):
        for lv in comp["levels"]:
            if lv.get("match_weight") is not None:
                rows.append([
                    comp["field"],
                    (lv["label"][:38] + "...") if len(lv["label"]) > 41 else lv["label"],
                    f"{lv['m_prob']:.4f}"         if lv["m_prob"] else "N/A",
                    f"{lv['u_prob']:.6f}"         if lv["u_prob"] else "N/A",
                    f"{lv['match_weight']:.4f}",
                ])
    if rows:
        pdf.table(
            ["Field", "Level", "m-prob", "u-prob", "Match weight"],
            rows,
            [CONTENT*0.12, CONTENT*0.36, CONTENT*0.14, CONTENT*0.18, CONTENT*0.20],
        )

    # Match weights chart
    mw = chart_match_weights(model_params)
    if mw:
        pdf.add_page()
        pdf.h2("Match Weights Chart")
        pdf.img(mw, "Figure: Match weights per comparison level. "
                "Bars right of zero increase match probability.")

    # Parameter estimates chart
    pe = chart_parameter_estimates(model_params)
    if pe:
        pdf.h2("Parameter Estimates Chart")
        pdf.img(pe, "Figure: m-probabilities as log odds per level "
                "estimated from EM training sessions.")


def _unlinkables_section(pdf, unlinkables, linkage_type):
    """Section 5: Unlinkable records chart (probabilistic only)."""
    if linkage_type == "deterministic" or not unlinkables.get("thresholds"):
        return
    pdf.add_page()
    pdf.h1("Unlinkable Records")
    pdf.body(
        "Some records lack sufficient information to produce a match weight above "
        "a given threshold. This chart shows what percentage will be unlinkable "
        "at each threshold, guiding the choice of operating threshold."
    )
    png = chart_unlinkables(unlinkables)
    if png:
        pdf.img(png, "Figure: Percentage of unlinkable records vs match-weight threshold.")


def _edge_metrics_section(pdf, metrics, linkage_type):
    """Section 6: Edge metrics + match weight histogram."""
    pdf.add_page()
    pdf.h1("Edge Metrics")
    pdf.body(
        "This section shows the results of model inference of pairwise edges. "
        "Quality review insights are provided for the number of edges predicted, "
        "distribution of match weights, and match probabilities."
    )

    pdf.h2("Edge metrics overview:")
    pdf.kv("Number of predicted edges",
           f"{metrics.get('n_edges', 0):,}")
    pdf.kv("Number of distinct unique_ids with an edge",
           f"{metrics.get('n_unique_ids', 0):,}")

    prob_stats = metrics.get("match_prob_stats", pd.DataFrame())
    if not prob_stats.empty:
        pdf.h2("Match probability statistics")
        pdf.table(
            ["Statistic", "Value"],
            [["Mean",    prob_stats["mean_match_prob"].iloc[0]],
             ["Median",  prob_stats["median_match_prob"].iloc[0]],
             ["Min",     prob_stats["min_match_prob"].iloc[0]],
             ["Max",     prob_stats["max_match_prob"].iloc[0]],
             ["Std Dev", prob_stats["stddev_match_prob"].iloc[0]]],
            [CONTENT * 0.5, CONTENT * 0.5],
        )

    # Match weight histogram
    wd = metrics.get("weight_dist", pd.DataFrame())
    if not wd.empty and len(wd) > 1:
        pdf.h2("Match Weight Histogram")
        png = chart_match_weight_histogram(wd)
        if png:
            pdf.img(png, "Figure: Histogram of match weights. "
                    "Peaks near high positive values indicate confident predictions.")

    # Gamma scores
    g = metrics.get("gamma_means", pd.DataFrame())
    if not g.empty and linkage_type == "probabilistic":
        pdf.h2("Mean gamma scores by field")
        g_rows = [[c.replace("gamma_", ""), f"{v:.4f}"]
                  for c, v in g.iloc[0].items()]
        if g_rows:
            pdf.table(["Field", "Mean Gamma"], g_rows,
                      [CONTENT * 0.6, CONTENT * 0.4])


def _cluster_metrics_section(pdf, metrics, operation_mode):
    """Section 7: Cluster metrics table + dataset overlap Venn diagram."""
    pdf.add_page()
    pdf.h1("Cluster Metrics")
    pdf.body(
        "This section shows the cluster inference results. A threshold was applied "
        "to the predicted edges to form entity clusters using a connected-components "
        "algorithm. Each cluster ideally represents one real-world individual."
    )

    pdf.h2("Cluster metrics overview:")
    pdf.kv("Number of clusters", f"{metrics.get('n_clusters', 0):,}")
    pdf.kv("Cross-dataset clusters", f"{metrics.get('n_cross_dataset', 0):,}")

    # Singleton vs multi-record
    s = metrics.get("singleton_stats", pd.DataFrame())
    if not s.empty:
        pdf.table(list(s.columns), [list(r) for _, r in s.iterrows()])

    # Cluster size distribution chart
    cs = metrics.get("cluster_sizes", pd.DataFrame())
    cs_png = chart_cluster_sizes(cs)
    if cs_png:
        pdf.img(cs_png, "Figure: Cluster size distribution.")

    # Dataset overlap Venn diagram
    venn = metrics.get("venn", {})
    if operation_mode != "dedupe":
        pdf.add_page()
        pdf.h2("Dataset overlap Venn diagram")
        pdf.body(
            "The Venn diagram shows how many clusters contain records from "
            "Dataset A only, Dataset B only, or both datasets. "
            "Cross-dataset clusters represent successfully linked records."
        )
        # Summary table
        a_only  = venn.get("a_only",  0)
        b_only  = venn.get("b_only",  0)
        both_ab = venn.get("both_ab", 0)
        pdf.table(
            ["Category", "N clusters"],
            [["A only", a_only], ["B only", b_only], ["Both A and B", both_ab]],
            [CONTENT * 0.7, CONTENT * 0.3],
        )
        venn_png = chart_venn_diagram(venn, operation_mode)
        if venn_png:
            pdf.img(venn_png, "Figure: Venn diagram of cluster_id set membership.")


def _confusion_section(pdf, cm: dict, truth_space_df: pd.DataFrame, crl: dict, lt: str):
    """Section 8: Confusion matrix + Precision-Recall curve + CRL score."""
    pdf.add_page()
    pdf.h1("Model Accuracy: Confusion Matrix")
    pdf.body(
        "Ground truth: the 'cluster' column in the input datasets. "
        "Records with the same cluster value are true matches. "
        "TP = predicted match AND true match. "
        "FP = predicted match BUT not true match. "
        "FN = true match NOT predicted. "
        "TN is omitted (too large for pairwise comparison)."
    )

    # Guard: return early if matrix is unavailable or errored
    if not cm or "error" in cm or cm.get("unavailable"):
        reason = (cm.get("unavailable_reason")
                  or cm.get("error")
                  or "Confusion matrix not available for this dataset.")
        pdf.body(reason)
        return

    # Guard None values — keys exist but may be None when cluster col is absent
    tp = cm.get("tp") or 0
    fp = cm.get("fp") or 0
    fn = cm.get("fn") or 0
    pdf.kv("True Positives (TP)",  f"{tp:,}")
    pdf.kv("False Positives (FP)", f"{fp:,}")
    pdf.kv("False Negatives (FN)", f"{fn:,}")
    pdf.kv("Ground truth pairs",   f"{cm.get('n_gt_edges') or 0:,}")
    pdf.kv("Predicted pairs",      f"{cm.get('n_pred_edges') or 0:,}")
    pdf.ln(2)

    pdf.h2("Derived Metrics")
    pdf.table(
        ["Metric", "Value", "Interpretation"],
        [
            ["Precision",  f"{cm.get('precision', 0):.4f}", "TP / (TP+FP)  - of predicted matches, how many are correct"],
            ["Recall",     f"{cm.get('recall', 0):.4f}",    "TP / (TP+FN)  - of true matches, how many were found"],
            ["F1 Score",   f"{cm.get('f1', 0):.4f}",        "Harmonic mean of Precision and Recall"],
            ["F* Score",   f"{cm.get('fstar', 0):.4f}",     "TP / (TP+FP+FN)  - accounts for both error types"],
            ["FDR",        f"{cm.get('fdr', 0):.4f}",       "False Discovery Rate: FP / (TP+FP)"],
            ["FNR",        f"{cm.get('fnr', 0):.4f}",       "False Negative Rate: FN / (TP+FN)"],
        ],
        [CONTENT * 0.18, CONTENT * 0.18, CONTENT * 0.64],
    )

    cm_png = chart_confusion_matrix(cm)
    if cm_png:
        pdf.img(cm_png, "Figure: Confusion matrix (pairwise). "
                "TN omitted as it is too large to count practically.")

    # Precision-Recall curve (probabilistic only)
    if lt == "probabilistic" and not truth_space_df.empty:
        pr_png = chart_precision_recall(truth_space_df)
        if pr_png:
            pdf.add_page()
            pdf.h2("Precision-Recall Curve and F* Score (example 16 - CRL analysis)")
            pdf.body(
                "These curves show Precision and Recall at every match-probability "
                "threshold. The F* curve shows the composite score TP/(TP+FP+FN). "
                "The ideal operating point maximises both Precision and Recall."
            )
            pdf.img(pr_png, "Figure: Precision-Recall curve (left) and F* vs threshold (right).")

        # CRL score
        if crl.get("crl_score") is not None:
            pdf.h2("CRL Score (Composite Reliability of Linkage)")
            pdf.body(
                "The CRL score measures linkage quality in the valid operating region "
                "where both FDR (False Discovery Rate) and FNR (False Negative Rate) "
                "are within the acceptable epsilon threshold. "
                "CRL = AVG(F*) * (t_upper - t_lower). "
                "A higher CRL score indicates more reliable linkage."
            )
            pdf.kv("Epsilon (tolerance)", f"{crl.get('epsilon', 0.1):.2f}")
            pdf.kv("CRL Score",           f"{crl.get('crl_score', 0):.6f}")
            pdf.kv("t_upper",             f"{crl.get('t_upper', 'N/A')}")
            pdf.kv("t_lower",             f"{crl.get('t_lower', 'N/A')}")
            pdf.kv("epsilon_z",           f"{crl.get('epsilon_z', 'N/A')}")


# =============================================================================
# ── PUBLIC API ────────────────────────────────────────────────────────────────
# =============================================================================

def generate_report(
    run_label:        str,
    run_config:       dict,
    metrics:          dict,
    n_input_records:  int,
    model_params:     Optional[dict] = None,
    missingness_a:    Optional[dict] = None,
    missingness_b:    Optional[dict] = None,
    blocking_counts:  Optional[list] = None,
    unlinkables:      Optional[dict] = None,
    settings_used:    Optional[dict] = None,
    confusion_matrix: Optional[dict] = None,
    truth_space_df:   Optional[pd.DataFrame] = None,
    crl_score:        Optional[dict] = None,
) -> bytes:
    """Generate the complete PDF Report.

    Sections in order:
      1. Cover page
      2. Dataset information + completeness chart
      3. Blocking rules + cumulative blocking chart
      4. Comparison methods
      5. Model training (probabilistic only)
         a. Match weights chart
         b. Parameter estimates chart
      6. Unlinkable records (probabilistic only)
      7. Edge metrics + match weight histogram
      8. Cluster metrics + Venn diagram
      9. Confusion matrix + Precision-Recall curve + CRL score
    """
    model_params     = model_params     or {}
    missingness_a    = missingness_a    or {}
    missingness_b    = missingness_b    or {}
    blocking_counts  = blocking_counts  or []
    unlinkables      = unlinkables      or {}
    settings_used    = settings_used    or {}
    confusion_matrix = confusion_matrix or {}
    truth_space_df   = truth_space_df   if truth_space_df is not None else pd.DataFrame()
    crl_score        = crl_score        or {}

    op_mode = run_config.get("operation_mode", "dedupe")
    lt      = run_config.get("linkage_type", "deterministic")
    fields  = run_config.get("selected_fields", [])

    pdf = _LinkageReport(run_label=run_label)

    _cover(pdf, run_config, run_label)
    _datasets_section(pdf, n_input_records, op_mode, fields, missingness_a, missingness_b)
    _blocking_section(pdf, run_config, blocking_counts)
    _comparison_methods_section(pdf, settings_used)
    if lt == "probabilistic":
        _model_training_section(pdf, run_config, fields)
        if model_params.get("training_complete"):
            _trained_model_section(pdf, model_params)
    _unlinkables_section(pdf, unlinkables, lt)
    _edge_metrics_section(pdf, metrics, lt)
    _cluster_metrics_section(pdf, metrics, op_mode)
    if confusion_matrix:
        _confusion_section(pdf, confusion_matrix, truth_space_df, crl_score, lt)

    return bytes(pdf.output())
