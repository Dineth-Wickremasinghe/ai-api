"""
analysis.py
-----------
Error analysis for the Fabric AI waste prediction model.
Table  : predictions
Columns: actual_wastage_pct, predicted_wastage_pct, Pattern_Complexity,
         Operator_Experience_Years, Fabric_Type_encoded,
         Cutting_Method_Manual, Fabric_Pattern_encoded, Marker_Loss_pct

BEFORE RUNNING:
  1. Run this SQL in pgAdmin to add the missing column (only once):
         ALTER TABLE predictions ADD COLUMN IF NOT EXISTS predicted_wastage_pct FLOAT;
  2. Make sure your API is saving predicted_wastage_pct when it makes predictions.

Run:
    python analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sqlalchemy import text
from database import engine


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

    mean_res = df["residual"].mean()
    std_res  = df["residual"].std()
    skew     = stats.skew(df["residual"].dropna())

    print(f"  Mean residual    : {mean_res:+.4f}  {'⚠ model over-predicts' if mean_res > 0 else '⚠ model under-predicts'}")
    print(f"  Std of residuals : {std_res:.4f}")
    print(f"  Skewness         : {skew:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Method 1 — Residual Analysis (Fabric Waste Model)", fontsize=13, fontweight="bold")

    axes[0].hist(df["residual"].dropna(), bins=40, color="#5B8FD4", edgecolor="white", linewidth=0.4)
    axes[0].axvline(0,        color="black", linestyle="--", linewidth=1.5, label="Zero error")
    axes[0].axvline(mean_res, color="red",   linestyle="--", linewidth=1.5, label=f"Mean ({mean_res:+.3f})")
    axes[0].set_xlabel("Residual  (predicted − actual wastage %)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of residuals")
    axes[0].legend()

    axes[1].scatter(df["predicted_wastage_pct"], df["residual"], alpha=0.35, s=18, color="#5B8FD4")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted wastage (%)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted  (funnel = heteroscedasticity)")

    plt.tight_layout()
    plt.savefig("plot_1_residuals.png", dpi=150)
    plt.show()
    print("  → Saved: plot_1_residuals.png")


# ── 3. Error Segmentation ──────────────────────────────────────────────────────

def error_segmentation(df):
    print("\n─── METHOD 2: ERROR SEGMENTATION ───")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Method 2 — Error Segmentation (Fabric Waste Model)", fontsize=13, fontweight="bold")

    # A. Pattern Complexity buckets
    if df["Pattern_Complexity"].notna().any():
        df["complex_bin"] = pd.cut(df["Pattern_Complexity"], bins=5)
        seg = df.groupby("complex_bin", observed=True)["error_pct"].mean()
        colors = ["#d94c4c" if v > 5 else "#5B8FD4" for v in seg.values]
        axes[0, 0].bar(range(len(seg)), seg.values, color=colors, edgecolor="white")
        axes[0, 0].set_xticks(range(len(seg)))
        axes[0, 0].set_xticklabels([str(i) for i in seg.index], rotation=25, ha="right", fontsize=8)
        axes[0, 0].axhline(0, color="black", linewidth=1)
        axes[0, 0].set_ylabel("Mean % error")
        axes[0, 0].set_title("Error by Pattern Complexity  (red = high bias)")

    # B. Operator Experience buckets
    if df["Operator_Experience_Years"].notna().any():
        df["exp_bin"] = pd.cut(df["Operator_Experience_Years"], bins=5)
        seg = df.groupby("exp_bin", observed=True)["error_pct"].mean()
        colors = ["#d94c4c" if v > 5 else "#5B8FD4" for v in seg.values]
        axes[0, 1].bar(range(len(seg)), seg.values, color=colors, edgecolor="white")
        axes[0, 1].set_xticks(range(len(seg)))
        axes[0, 1].set_xticklabels([str(i) for i in seg.index], rotation=25, ha="right", fontsize=8)
        axes[0, 1].axhline(0, color="black", linewidth=1)
        axes[0, 1].set_ylabel("Mean % error")
        axes[0, 1].set_title("Error by Operator Experience (Years)")

    # C. Fabric Type (encoded integer — categorical)
    if df["Fabric_Type_encoded"].notna().any():
        seg = df.groupby("Fabric_Type_encoded")["error_pct"].mean().sort_values()
        colors = ["#d94c4c" if v > 5 else "#5B8FD4" for v in seg.values]
        axes[1, 0].bar([str(i) for i in seg.index], seg.values, color=colors, edgecolor="white")
        axes[1, 0].axhline(0, color="black", linewidth=1)
        axes[1, 0].set_xlabel("Fabric Type (encoded)")
        axes[1, 0].set_ylabel("Mean % error")
        axes[1, 0].set_title("Error by Fabric Type")

    # D. Cutting Method Manual (0 = automated, 1 = manual)
    if df["Cutting_Method_Manual"].notna().any():
        seg = df.groupby("Cutting_Method_Manual")["error_pct"].mean()
        labels = {0: "Automated", 1: "Manual"}
        colors = ["#d94c4c" if v > 5 else "#5B8FD4" for v in seg.values]
        axes[1, 1].bar([labels.get(i, str(i)) for i in seg.index], seg.values, color=colors, edgecolor="white")
        axes[1, 1].axhline(0, color="black", linewidth=1)
        axes[1, 1].set_ylabel("Mean % error")
        axes[1, 1].set_title("Error by Cutting Method")

    plt.tight_layout()
    plt.savefig("plot_2_segmentation.png", dpi=150)
    plt.show()
    print("  → Saved: plot_2_segmentation.png")


# ── 4. Z-Score Flagging ────────────────────────────────────────────────────────

def zscore_flagging(df, threshold=2.5):
    print(f"\n─── METHOD 3: Z-SCORE FLAGGING  (threshold = ±{threshold}σ) ───")

    mean = df["error_pct"].mean()
    std  = df["error_pct"].std()

    df["z_score"] = (df["error_pct"] - mean) / std
    flagged = df[df["z_score"].abs() > threshold].copy()

    print(f"  Total rows  : {len(df)}")
    print(f"  Flagged     : {len(flagged)}  ({100 * len(flagged) / len(df):.1f}%)")

    if not flagged.empty:
        print("\n  Top 10 worst predictions:")
        cols = ["id", "Fabric_Type_encoded", "Cutting_Method_Manual",
                "actual_wastage_pct", "predicted_wastage_pct", "error_pct", "z_score"]
        print(flagged.nlargest(10, "z_score")[cols].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Method 3 — Z-Score Flagging (Fabric Waste Model)", fontsize=13, fontweight="bold")

    normal = df[df["z_score"].abs() <= threshold]
    axes[0].scatter(normal["predicted_wastage_pct"],  normal["error_pct"],  alpha=0.3, s=14, color="#5B8FD4", label="Normal")
    axes[0].scatter(flagged["predicted_wastage_pct"], flagged["error_pct"], alpha=0.8, s=40, color="#d94c4c", label=f"Flagged (|z|>{threshold})")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Predicted wastage (%)")
    axes[0].set_ylabel("% error")
    axes[0].set_title("Flagged predictions (red dots)")
    axes[0].legend()

    axes[1].hist(df["z_score"].dropna(), bins=40, color="#5B8FD4", edgecolor="white", linewidth=0.4)
    axes[1].axvline( threshold, color="red", linestyle="--", linewidth=1.5, label=f"+{threshold}σ")
    axes[1].axvline(-threshold, color="red", linestyle="--", linewidth=1.5, label=f"-{threshold}σ")
    axes[1].set_xlabel("Z-score of % error")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Z-score distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("plot_3_zscore.png", dpi=150)
    plt.show()
    print("  → Saved: plot_3_zscore.png")

    return flagged


# ── 5. Drift Detection ─────────────────────────────────────────────────────────

def drift_detection(df, window=50):
    print(f"\n─── METHOD 4: DRIFT DETECTION  (window = {window} rows) ───")

    df = df.sort_values("created_at").reset_index(drop=True)
    df["rolling_mean_err"] = df["error_abs"].rolling(window=window).mean()

    valid = df["rolling_mean_err"].dropna()
    if len(valid) < 2:
        print(f"  Not enough rows yet — need at least {window} rows with both values filled.")
        return

    trend = valid.iloc[-1] - valid.iloc[0]
    print(f"  First window mean absolute error : {valid.iloc[0]:.4f}")
    print(f"  Last  window mean absolute error : {valid.iloc[-1]:.4f}")
    print(f"  Trend                            : {trend:+.4f}  {'⚠ model degrading' if trend > 0.5 else '✓ stable'}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Method 4 — Drift Detection (Fabric Waste Model)", fontsize=13, fontweight="bold")

    axes[0].scatter(df.index, df["error_pct"], alpha=0.25, s=10, color="#5B8FD4")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("% error")
    axes[0].set_title("Raw % error per prediction (ordered by created_at)")

    axes[1].plot(df.index, df["rolling_mean_err"], color="#d94c4c", linewidth=2,
                 label=f"Rolling mean abs error (window={window})")
    axes[1].fill_between(df.index, df["rolling_mean_err"], alpha=0.15, color="#d94c4c")
    axes[1].set_xlabel("Row index (time order)")
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_title("Rising trend = model drift / degradation")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("plot_4_drift.png", dpi=150)
    plt.show()
    print("  → Saved: plot_4_drift.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    if df.empty:
        print("\n⚠ No rows found with both actual_wastage_pct AND predicted_wastage_pct filled.")
        print("  Steps to fix:")
        print("  1. Run in pgAdmin:  ALTER TABLE predictions ADD COLUMN IF NOT EXISTS predicted_wastage_pct FLOAT;")
        print("  2. Make sure your API saves predicted_wastage_pct each time it predicts.")
    else:
        residual_analysis(df)
        error_segmentation(df)
        zscore_flagging(df)
        drift_detection(df)
        print("\n✓ All analyses complete. PNG files saved in your project folder.")