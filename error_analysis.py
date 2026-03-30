import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy import stats
from sqlalchemy import text
from database import engine


# ── Palette ────────────────────────────────────────────────────────────────────

BLUE       = "#378ADD"
BLUE_LIGHT = "#B5D4F4"
BLUE_FAINT = "#E6F1FB"
RED        = "#E24B4A"
RED_LIGHT  = "#F7C1C1"
AMBER      = "#EF9F27"
TEAL       = "#1D9E75"
GRAY       = "#888780"
GRAY_LIGHT = "#D3D1C7"
BG         = "#F8F8F6"
WHITE      = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    WHITE,
    "axes.edgecolor":    GRAY_LIGHT,
    "axes.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        GRAY_LIGHT,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.6,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "sans-serif",
    "text.color":        "#2C2C2A",
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bar_colors(values, threshold=5.0):
    """Return bar colors: red when mean % error > threshold, else blue."""
    return [RED if v > threshold else BLUE for v in values]


def _metric_card(ax, value_str, label, sub="", color="#2C2C2A"):
    """Draw a summary metric card on a dedicated axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch((0.04, 0.08), 0.92, 0.84,
                         boxstyle="round,pad=0.02",
                         facecolor=BG, edgecolor=GRAY_LIGHT, linewidth=0.8)
    ax.add_patch(box)
    ax.text(0.5, 0.68, value_str, ha="center", va="center",
            fontsize=17, fontweight="bold", color=color)
    ax.text(0.5, 0.38, label, ha="center", va="center",
            fontsize=9, color=GRAY)
    if sub:
        ax.text(0.5, 0.18, sub, ha="center", va="center",
                fontsize=8, color=GRAY_LIGHT)


def _section_title(ax, text_str):
    ax.text(0, 1.08, text_str, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#2C2C2A",
            va="bottom")


# ── 1. Load from predictions table ────────────────────────────────────────────

def load_data():
    query = """
        SELECT
            id,
            created_at,
            actual_wastage_pct,
            predicted_wastage_pct,
            "Pattern_Complexity",
            "Operator_Experience_Years",
            "Fabric_Pattern_encoded",
            "Cutting_Method_Manual",
            "Fabric_Type_encoded",
            "Marker_Loss_pct"
        FROM predictions
        WHERE actual_wastage_pct    IS NOT NULL
          AND predicted_wastage_pct IS NOT NULL
        ORDER BY created_at
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    df["residual"]  = df["predicted_wastage_pct"] - df["actual_wastage_pct"]
    df["error_abs"] = df["residual"].abs()
    df["error_pct"] = (df["residual"] / df["actual_wastage_pct"].replace(0, np.nan)) * 100

    print(f"Loaded {len(df)} rows with both actual and predicted values.")
    return df


# ── 2. Residual Analysis ───────────────────────────────────────────────────────

def residual_analysis(df):
    print("\n─── METHOD 1: RESIDUAL ANALYSIS ───")

    residuals = df["residual"].dropna()
    mean_res  = residuals.mean()
    std_res   = residuals.std()
    skew      = stats.skew(residuals)
    kurt      = stats.kurtosis(residuals)
    n         = len(residuals)

    print(f"  Mean residual    : {mean_res:+.4f}  "
          f"{'⚠ model over-predicts' if mean_res > 0 else '⚠ model under-predicts'}")
    print(f"  Std of residuals : {std_res:.4f}")
    print(f"  Skewness         : {skew:+.3f}")
    print(f"  Kurtosis         : {kurt:+.3f}")

    fig = plt.figure(figsize=(15, 11), facecolor=BG)
    fig.suptitle("Method 1 — Residual Analysis  ·  Fabric Waste Model",
                 fontsize=13, fontweight="bold", y=0.97, color="#2C2C2A")

    # ── metric cards row ──────────────────────────────────────────────────────
    gs_top = gridspec.GridSpec(1, 4, figure=fig,
                               left=0.04, right=0.98, top=0.88, bottom=0.72,
                               wspace=0.08)
    bias_color = RED if abs(mean_res) > 0.3 else TEAL
    _metric_card(fig.add_subplot(gs_top[0]), f"{mean_res:+.3f}", "Mean residual",
                 "over-predicts" if mean_res > 0 else "under-predicts", bias_color)
    _metric_card(fig.add_subplot(gs_top[1]), f"{std_res:.3f}", "Std of residuals", "spread around zero")
    _metric_card(fig.add_subplot(gs_top[2]), f"{skew:+.3f}", "Skewness",
                 "right tail heavy" if skew > 0.5 else ("left tail heavy" if skew < -0.5 else "approx. symmetric"))
    _metric_card(fig.add_subplot(gs_top[3]), f"{kurt:+.3f}", "Excess kurtosis",
                 "heavier tails than normal" if kurt > 0 else "lighter tails")

    # ── charts row ────────────────────────────────────────────────────────────
    gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.06, right=0.98, top=0.67, bottom=0.08,
                               wspace=0.22)

    # Chart A — residual histogram with normal overlay
    ax1 = fig.add_subplot(gs_bot[0])
    _section_title(ax1, "A  Distribution of residuals")
    ax1.hist(residuals, bins=45, color=BLUE, alpha=0.85,
             edgecolor=WHITE, linewidth=0.3, density=True, label="Residuals")

    x_fit = np.linspace(residuals.min(), residuals.max(), 300)
    ax1.plot(x_fit, stats.norm.pdf(x_fit, mean_res, std_res),
             color=BLUE_LIGHT, linewidth=2, linestyle="--", label="Normal fit")

    ax1.axvline(0,        color="#2C2C2A", linewidth=1.2,
                linestyle="--", label="Zero error")
    ax1.axvline(mean_res, color=RED, linewidth=1.8,
                linestyle="-",  label=f"Mean ({mean_res:+.3f})")
    ax1.axvspan(mean_res - std_res, mean_res + std_res,
                alpha=0.07, color=RED, label="±1σ band")

    ax1.set_xlabel("Residual  (predicted − actual wastage %)", fontsize=9)
    ax1.set_ylabel("Density", fontsize=9)
    ax1.legend(fontsize=8, framealpha=0.5)
    ax1.yaxis.grid(True)

    # Chart B — residuals vs predicted (heteroscedasticity)
    ax2 = fig.add_subplot(gs_bot[1])
    _section_title(ax2, "B  Residuals vs predicted  (funnel = heteroscedasticity)")

    sc = ax2.scatter(df["predicted_wastage_pct"], df["residual"],
                     alpha=0.35, s=18, c=df["error_abs"],
                     cmap="RdYlBu_r", vmin=0, vmax=df["error_abs"].quantile(0.95))
    cb = fig.colorbar(sc, ax=ax2, pad=0.02, shrink=0.85)
    cb.set_label("Abs error", fontsize=8)
    cb.ax.tick_params(labelsize=8)

    ax2.axhline(0, color=RED, linestyle="--", linewidth=1.5)

    # LOWESS smoothed trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_idx = df["predicted_wastage_pct"].argsort()
        px = df["predicted_wastage_pct"].values[sorted_idx]
        ry = df["residual"].values[sorted_idx]
        sm = lowess(ry, px, frac=0.3, return_sorted=True)
        ax2.plot(sm[:, 0], sm[:, 1], color=AMBER, linewidth=2,
                 label="LOWESS trend", zorder=5)
        ax2.legend(fontsize=8, framealpha=0.5)
    except ImportError:
        pass

    ax2.set_xlabel("Predicted wastage (%)", fontsize=9)
    ax2.set_ylabel("Residual", fontsize=9)
    ax2.yaxis.grid(True)

    plt.savefig("plot_1_residuals.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Saved: plot_1_residuals.png")


# ── 3. Error Segmentation ──────────────────────────────────────────────────────

def error_segmentation(df):
    print("\n─── METHOD 2: ERROR SEGMENTATION ───")

    fig = plt.figure(figsize=(15, 13), facecolor=BG)
    fig.suptitle("Method 2 — Error Segmentation  ·  Fabric Waste Model",
                 fontsize=13, fontweight="bold", y=0.97, color="#2C2C2A")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.07, right=0.97, top=0.90, bottom=0.06,
                           wspace=0.30, hspace=0.45)

    def _seg_bar(ax, seg_series, title, xlabel, threshold=5.0, horizontal=False):
        seg_series = seg_series.dropna().sort_values()
        labels     = [str(i) for i in seg_series.index]
        values     = seg_series.values
        colors     = _bar_colors(values, threshold)

        if horizontal:
            bars = ax.barh(labels, values, color=colors, edgecolor=WHITE,
                           linewidth=0.4, height=0.55)
            ax.axvline(0, color="#2C2C2A", linewidth=0.8)
            ax.axvline(threshold, color=RED, linewidth=1, linestyle="--", alpha=0.6)
            for bar, val in zip(bars, values):
                ax.text(val + 0.15, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=8, color="#2C2C2A")
            ax.set_xlabel("Mean % error", fontsize=9)
        else:
            bars = ax.bar(labels, values, color=colors, edgecolor=WHITE,
                          linewidth=0.4, width=0.55)
            ax.axhline(0, color="#2C2C2A", linewidth=0.8)
            ax.axhline(threshold, color=RED, linewidth=1, linestyle="--", alpha=0.6)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                        f"{val:.1f}%", ha="center", fontsize=8, color="#2C2C2A")
            ax.set_ylabel("Mean % error", fontsize=9)

        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=9) if not horizontal else None
        ax.tick_params(axis="x", rotation=20 if not horizontal else 0)
        ax.yaxis.grid(True)
        legend_els = [
            Line2D([0], [0], color=RED, linewidth=1.2, linestyle="--",
                   label=f"High bias threshold ({threshold}%)"),
            plt.Rectangle((0, 0), 1, 1, fc=RED,   label="High bias  (>5%)"),
            plt.Rectangle((0, 0), 1, 1, fc=BLUE,  label="Acceptable (<5%)"),
        ]
        ax.legend(handles=legend_els, fontsize=7.5, framealpha=0.5)

    # A. Pattern Complexity
    if df["Pattern_Complexity"].notna().any():
        df["complex_bin"] = pd.cut(df["Pattern_Complexity"], bins=5)
        seg = df.groupby("complex_bin", observed=True)["error_pct"].mean()
        _seg_bar(fig.add_subplot(gs[0, 0]), seg,
                 "A  Error by pattern complexity", "Pattern complexity bucket")

    # B. Operator Experience
    if df["Operator_Experience_Years"].notna().any():
        df["exp_bin"] = pd.cut(df["Operator_Experience_Years"], bins=5)
        seg = df.groupby("exp_bin", observed=True)["error_pct"].mean()
        _seg_bar(fig.add_subplot(gs[0, 1]), seg,
                 "B  Error by operator experience", "Experience bucket (years)")

    # C. Fabric Type — horizontal to fit encoded labels
    if df["Fabric_Type_encoded"].notna().any():
        seg = df.groupby("Fabric_Type_encoded")["error_pct"].mean()
        _seg_bar(fig.add_subplot(gs[1, 0]), seg,
                 "C  Error by fabric type (encoded)", "Fabric type",
                 horizontal=True)

    # D. Cutting Method with box-plot overlay
    if df["Cutting_Method_Manual"].notna().any():
        ax_d = fig.add_subplot(gs[1, 1])
        labels_cm = {0: "Automated", 1: "Manual"}
        groups    = [df[df["Cutting_Method_Manual"] == k]["error_pct"].dropna()
                     for k in sorted(labels_cm)]
        seg       = df.groupby("Cutting_Method_Manual")["error_pct"].mean()
        xpos      = range(len(seg))
        colors_cm = _bar_colors(seg.values)

        ax_d.bar(xpos, seg.values, color=colors_cm,
                 edgecolor=WHITE, linewidth=0.4, width=0.45, alpha=0.8)
        ax_d.boxplot(groups, positions=list(xpos), widths=0.25,
                     patch_artist=False,
                     medianprops=dict(color=AMBER, linewidth=2),
                     whiskerprops=dict(linewidth=0.8),
                     capprops=dict(linewidth=0.8),
                     flierprops=dict(marker="o", markersize=3,
                                     markerfacecolor=GRAY, alpha=0.4))
        ax_d.axhline(0, color="#2C2C2A", linewidth=0.8)
        ax_d.axhline(5, color=RED, linewidth=1, linestyle="--", alpha=0.6)
        ax_d.set_xticks(list(xpos))
        ax_d.set_xticklabels([labels_cm[k] for k in sorted(labels_cm)], fontsize=9)
        ax_d.set_ylabel("Mean % error", fontsize=9)
        ax_d.set_title("D  Error by cutting method  (+ distribution box)", fontsize=10,
                       fontweight="bold", pad=8)
        ax_d.yaxis.grid(True)

    plt.savefig("plot_2_segmentation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Saved: plot_2_segmentation.png")


# ── 4. Z-Score Flagging ────────────────────────────────────────────────────────

def zscore_flagging(df, threshold=2.5):
    print(f"\n─── METHOD 3: Z-SCORE FLAGGING  (threshold = ±{threshold}σ) ───")

    mean_e  = df["error_pct"].mean()
    std_e   = df["error_pct"].std()
    df      = df.copy()
    df["z_score"] = (df["error_pct"] - mean_e) / std_e
    flagged = df[df["z_score"].abs() > threshold].copy()

    pct_flagged = 100 * len(flagged) / len(df)
    print(f"  Total rows  : {len(df)}")
    print(f"  Flagged     : {len(flagged)}  ({pct_flagged:.1f}%)")

    if not flagged.empty:
        print("\n  Top 10 worst predictions:")
        cols = ["id", "Fabric_Type_encoded", "Cutting_Method_Manual",
                "actual_wastage_pct", "predicted_wastage_pct",
                "error_pct", "z_score"]
        print(flagged.nlargest(10, "z_score")[cols].to_string(index=False))

    fig = plt.figure(figsize=(15, 13), facecolor=BG)
    fig.suptitle(f"Method 3 — Z-Score Flagging  (threshold ±{threshold}σ)  ·  Fabric Waste Model",
                 fontsize=13, fontweight="bold", y=0.97, color="#2C2C2A")

    # ── metric cards ──────────────────────────────────────────────────────────
    gs_top = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.04, right=0.98, top=0.88, bottom=0.74,
                               wspace=0.08)
    _metric_card(fig.add_subplot(gs_top[0]),
                 f"{len(df)}", "Total predictions", "rows with both values")
    _metric_card(fig.add_subplot(gs_top[1]),
                 f"{len(flagged)}", "Flagged outliers",
                 f"|z| > {threshold}σ", RED if pct_flagged > 5 else TEAL)
    _metric_card(fig.add_subplot(gs_top[2]),
                 f"{pct_flagged:.1f}%", "Outlier rate",
                 "high" if pct_flagged > 8 else "acceptable",
                 RED if pct_flagged > 8 else TEAL)

    # ── charts ────────────────────────────────────────────────────────────────
    gs_bot = gridspec.GridSpec(2, 2, figure=fig,
                               left=0.07, right=0.97, top=0.70, bottom=0.06,
                               wspace=0.28, hspace=0.40)

    normal_mask = df["z_score"].abs() <= threshold

    # Chart A — scatter flagged vs normal
    ax1 = fig.add_subplot(gs_bot[0, 0])
    _section_title(ax1, "A  Flagged vs normal predictions")
    ax1.scatter(df.loc[normal_mask, "predicted_wastage_pct"],
                df.loc[normal_mask, "error_pct"],
                alpha=0.25, s=14, color=BLUE, label="Normal")
    ax1.scatter(df.loc[~normal_mask, "predicted_wastage_pct"],
                df.loc[~normal_mask, "error_pct"],
                alpha=0.85, s=55, color=RED, zorder=5,
                edgecolors=WHITE, linewidths=0.6,
                label=f"Flagged  (|z|>{threshold})")
    ax1.axhline(0, color="#2C2C2A", linewidth=0.8)
    ax1.set_xlabel("Predicted wastage (%)", fontsize=9)
    ax1.set_ylabel("% error", fontsize=9)
    ax1.legend(fontsize=8, framealpha=0.5)
    ax1.yaxis.grid(True)

    # Chart B — z-score histogram
    ax2 = fig.add_subplot(gs_bot[0, 1])
    _section_title(ax2, "B  Z-score distribution")
    z_vals = df["z_score"].dropna()
    n_bins = 50
    bin_edges = np.linspace(z_vals.min(), z_vals.max(), n_bins + 1)
    counts, edges = np.histogram(z_vals, bins=bin_edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    bar_colors  = [RED if abs(c) > threshold else BLUE for c in bin_centers]
    ax2.bar(bin_centers, counts, width=(edges[1] - edges[0]) * 0.9,
            color=bar_colors, edgecolor=WHITE, linewidth=0.2)
    ax2.axvline( threshold, color=RED, linestyle="--", linewidth=1.5,
                label=f"+{threshold}σ")
    ax2.axvline(-threshold, color=RED, linestyle="--", linewidth=1.5,
                label=f"-{threshold}σ")
    x_fit = np.linspace(z_vals.min(), z_vals.max(), 300)
    ax2.plot(x_fit, stats.norm.pdf(x_fit) * len(z_vals) * (edges[1] - edges[0]),
             color=GRAY, linewidth=1.5, linestyle=":", label="Standard normal")
    ax2.set_xlabel("Z-score of % error", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.5)
    ax2.yaxis.grid(True)

    # Chart C — cumulative error distribution
    ax3 = fig.add_subplot(gs_bot[1, 0])
    _section_title(ax3, "C  Cumulative % error distribution")
    sorted_err = np.sort(df["error_pct"].dropna())
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax3.plot(sorted_err, cdf, color=BLUE, linewidth=1.8)
    ax3.axvline(np.percentile(sorted_err, 90), color=AMBER, linewidth=1.5,
                linestyle="--", label="P90")
    ax3.axvline(np.percentile(sorted_err, 95), color=RED, linewidth=1.5,
                linestyle="--", label="P95")
    ax3.set_xlabel("% error", fontsize=9)
    ax3.set_ylabel("Cumulative proportion", fontsize=9)
    ax3.legend(fontsize=8, framealpha=0.5)
    ax3.yaxis.grid(True)

    # Chart D — top-10 flagged sorted bar
    ax4 = fig.add_subplot(gs_bot[1, 1])
    _section_title(ax4, "D  Top-10 flagged — z-score magnitude")
    if not flagged.empty:
        top10 = flagged.nlargest(10, "z_score").reset_index(drop=True)
        bar_c = [RED if z > 3.5 else AMBER for z in top10["z_score"]]
        bars  = ax4.barh(range(len(top10)), top10["z_score"],
                         color=bar_c, edgecolor=WHITE, linewidth=0.4)
        ax4.set_yticks(range(len(top10)))
        ax4.set_yticklabels([f"ID {int(r['id'])}" for _, r in top10.iterrows()],
                            fontsize=8)
        ax4.axvline(threshold, color=RED, linewidth=1, linestyle="--", alpha=0.7)
        for bar, val in zip(bars, top10["z_score"]):
            ax4.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}σ", va="center", fontsize=8)
        ax4.set_xlabel("Z-score", fontsize=9)
        ax4.xaxis.grid(True)

    plt.savefig("plot_3_zscore.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Saved: plot_3_zscore.png")

    return flagged


# ── 5. Drift Detection ─────────────────────────────────────────────────────────

def drift_detection(df, window=50):
    print(f"\n─── METHOD 4: DRIFT DETECTION  (window = {window} rows) ───")

    df = df.sort_values("created_at").reset_index(drop=True)
    df["rolling_mean_err"] = df["error_abs"].rolling(window=window).mean()
    df["rolling_std_err"]  = df["error_abs"].rolling(window=window).std()
    df["rolling_med_err"]  = df["error_abs"].rolling(window=window).median()

    valid = df["rolling_mean_err"].dropna()
    if len(valid) < 2:
        print(f"  Not enough rows — need at least {window} rows with both values filled.")
        return

    first = valid.iloc[0]
    last  = valid.iloc[-1]
    trend = last - first
    print(f"  First window MAE : {first:.4f}")
    print(f"  Last  window MAE : {last:.4f}")
    print(f"  Trend Δ          : {trend:+.4f}  "
          f"{'⚠ model degrading' if trend > 0.5 else '✓ stable'}")

    fig = plt.figure(figsize=(15, 13), facecolor=BG)
    fig.suptitle(f"Method 4 — Drift Detection  (window = {window})  ·  Fabric Waste Model",
                 fontsize=13, fontweight="bold", y=0.97, color="#2C2C2A")

    # ── metric cards ──────────────────────────────────────────────────────────
    gs_top = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.04, right=0.98, top=0.88, bottom=0.74,
                               wspace=0.08)
    _metric_card(fig.add_subplot(gs_top[0]),
                 f"{first:.3f}", "First window MAE", f"rows 0–{window}")
    _metric_card(fig.add_subplot(gs_top[1]),
                 f"{last:.3f}", "Latest window MAE",
                 f"last {window} rows",
                 RED if trend > 0.5 else TEAL)
    trend_color = RED if trend > 0.5 else (AMBER if trend > 0.1 else TEAL)
    _metric_card(fig.add_subplot(gs_top[2]),
                 f"{trend:+.3f}", "Trend Δ (MAE)",
                 "degrading ⚠" if trend > 0.5 else "stable ✓",
                 trend_color)

    # ── charts ────────────────────────────────────────────────────────────────
    gs_bot = gridspec.GridSpec(3, 1, figure=fig,
                               left=0.07, right=0.97, top=0.70, bottom=0.06,
                               hspace=0.45)

    # Chart A — raw % error scatter
    ax1 = fig.add_subplot(gs_bot[0])
    _section_title(ax1, "A  Raw % error per prediction  (time order)")
    ax1.scatter(df.index, df["error_pct"],
                alpha=0.20, s=8, color=BLUE, rasterized=True)
    ax1.axhline(0, color="#2C2C2A", linewidth=0.8)
    ax1.set_ylabel("% error", fontsize=9)
    ax1.yaxis.grid(True)

    # Chart B — rolling mean + std band
    ax2 = fig.add_subplot(gs_bot[1])
    _section_title(ax2, "B  Rolling mean absolute error  — rising = drift")
    ax2.fill_between(df.index,
                     df["rolling_mean_err"] - df["rolling_std_err"],
                     df["rolling_mean_err"] + df["rolling_std_err"],
                     alpha=0.12, color=RED, label="±1 std band")
    ax2.plot(df.index, df["rolling_mean_err"], color=RED, linewidth=2,
             label=f"Rolling MAE (window={window})")
    ax2.plot(df.index, df["rolling_med_err"], color=AMBER, linewidth=1.2,
             linestyle="--", alpha=0.8, label="Rolling median")
    if trend > 0:
        valid_idx = df.index[df["rolling_mean_err"].notna()]
        z = np.polyfit(valid_idx, valid.values, 1)
        p = np.poly1d(z)
        ax2.plot(valid_idx, p(valid_idx), color=GRAY, linewidth=1,
                 linestyle=":", label="Linear trend")
    ax2.set_ylabel("MAE", fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.5)
    ax2.yaxis.grid(True)

    # Chart C — rolling error volatility (std)
    ax3 = fig.add_subplot(gs_bot[2])
    _section_title(ax3, "C  Rolling error std  — rising = increasing instability")
    ax3.fill_between(df.index, df["rolling_std_err"],
                     alpha=0.20, color=AMBER)
    ax3.plot(df.index, df["rolling_std_err"], color=AMBER, linewidth=1.8,
             label="Rolling std of abs error")
    ax3.set_xlabel("Row index  (time order)", fontsize=9)
    ax3.set_ylabel("Std", fontsize=9)
    ax3.legend(fontsize=8, framealpha=0.5)
    ax3.yaxis.grid(True)

    plt.savefig("plot_4_drift.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Saved: plot_4_drift.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    if df.empty:
        print("\n⚠ No rows found with both actual_wastage_pct AND predicted_wastage_pct filled.")
        print("  Steps to fix:")
        print("  1. Run in pgAdmin:")
        print("     ALTER TABLE predictions")
        print("     ADD COLUMN IF NOT EXISTS predicted_wastage_pct FLOAT;")
        print("  2. Make sure your API saves predicted_wastage_pct each time it predicts.")
    else:
        residual_analysis(df)
        error_segmentation(df)
        zscore_flagging(df)
        drift_detection(df)
        print("\n✓ All analyses complete. PNG files saved in your project folder.")