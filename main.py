# =============================================================================
# FINAL PROJECT: Engineering Data Systems Pipeline
# Course:        Computer Programming
# Academic Year: 2026
# Pillar:        PILLAR 4 — Manufacturing & Production
# Topic:         MAN-01: CNC Surface Roughness Analytics
# Student Name:  AISY JIBRIL B. CORDERO
# Student ID:    TUPM-25-0354
# Reporting Std: IEEE Two-Column Research Format
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
STUDENT_NAME = "AISY JIBRIL B. CORDERO"
STUDENT_ID   = "TUPM-25-0354"
TOPIC        = "MAN-01: CNC Surface Roughness Analytics"

DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"
RAW_FILE    = os.path.join(DATA_DIR, "dataset_original.csv")
CLEAN_FILE  = os.path.join(DATA_DIR, "dataset_cleaned.csv")

# Source files from Kaggle dataset (CNC turning: roughness, forces and tool wear)
SOURCE_FILES = [
    os.path.join(DATA_DIR, "Exp1.csv"),
    os.path.join(DATA_DIR, "Exp2.csv"),
    os.path.join(DATA_DIR, "Prep.csv"),
]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Unique filter: keep only records where feed rate (f) is in the lower 50th
# percentile — isolates fine-finish machining conditions unique to this analysis.
UNIQUE_FILTER_COL       = "f"          # feed rate column (adjust if CSV differs)
UNIQUE_FILTER_QUANTILE  = 0.50         # keep rows below this quantile


# =============================================================================
# CLASS: CNCPipeline
# Encapsulates all pipeline stages: Ingestion → Cleaning → Analysis → Visualize
# =============================================================================
class CNCPipeline:
    """
    Production-grade OOP pipeline for CNC Surface Roughness Analytics.
    Implements modular design, robust error handling, and NumPy-based statistics.
    """

    def __init__(self, raw_path: str, clean_path: str):
        self.raw_path   = raw_path
        self.clean_path = clean_path
        self.df_raw     = None   # original dataset
        self.df         = None   # cleaned / filtered dataset
        self.stats      = {}     # computed statistics store
        self.target_col = None   # surface roughness column name

    # ------------------------------------------------------------------
    # STAGE 1 — DATA INGESTION
    # ------------------------------------------------------------------
    def ingest(self) -> None:
        """Load the raw CSV dataset with error handling."""
        print("\n" + "="*65)
        print(f"  {TOPIC}")
        print(f"  Student: {STUDENT_NAME}  |  ID: {STUDENT_ID}")
        print("="*65)
        print("\n[STAGE 1] Data Ingestion...")

        try:
            if os.path.exists(self.raw_path):
                self.df_raw = pd.read_csv(self.raw_path)
                print(f"  ✔ Loaded existing '{self.raw_path}' — shape: {self.df_raw.shape}")
            else:
                frames = []
                for fpath in SOURCE_FILES:
                    try:
                        df_temp = pd.read_csv(fpath)
                        df_temp["source_file"] = os.path.basename(fpath)
                        frames.append(df_temp)
                        print(f"    ✔ Loaded: {fpath} — {df_temp.shape[0]} rows")
                    except Exception as e:
                        print(f"    ⚠ Skipped {fpath}: {e}")
                if not frames:
                    raise FileNotFoundError("No source CSV files found in data/ folder.")
                self.df_raw = pd.concat(frames, ignore_index=True)
                self.df_raw.to_csv(self.raw_path, index=False)
                print(f"  ✔ Merged {len(frames)} files → saved to '{self.raw_path}'")
            print(f"  Columns: {list(self.df_raw.columns)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        # Auto-detect the surface roughness (Ra) column
        ra_candidates = [c for c in self.df_raw.columns
                         if any(kw in c.lower() for kw in ["ra", "roughness", "surface"])]
        if ra_candidates:
            self.target_col = ra_candidates[0]
            print(f"  ✔ Target column detected: '{self.target_col}'")
        else:
            self.target_col = self.df_raw.select_dtypes(include=[np.number]).columns[-1]
            print(f"  ⚠ No 'Ra' column found — using last numeric column: '{self.target_col}'")

    # ------------------------------------------------------------------
    # STAGE 2 — DATA CLEANING & UNIQUE FILTER
    # ------------------------------------------------------------------
    def clean(self) -> None:
        """
        Automated pipeline:
          • Remove duplicates
          • Handle missing / null values
          • Correct corrupted data types
          • Apply unique programmatic filter (feed rate ≤ 50th percentile)
        """
        print("\n[STAGE 2] Data Cleaning & Filtering...")
        df = self.df_raw.copy()

        # 2a. Remove duplicates
        before = len(df)
        df.drop_duplicates(inplace=True)
        print(f"  Duplicates removed : {before - len(df)} rows")

        # 2b. Handle missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"  Null values detected:\n{null_counts[null_counts > 0]}")
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"    → '{col}' nulls filled with median ({median_val:.4f})")
        else:
            print("  ✔ No missing values detected.")

        # 2c. Correct corrupted data types
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass   # keep as-is if not numeric

        # 2d. Unique programmatic filter
        if UNIQUE_FILTER_COL in df.columns:
            threshold = df[UNIQUE_FILTER_COL].quantile(UNIQUE_FILTER_QUANTILE)
            before_filter = len(df)
            df = df[df[UNIQUE_FILTER_COL] <= threshold].copy()
            print(f"  Unique filter: kept {UNIQUE_FILTER_COL} ≤ {threshold:.4f} "
                  f"({before_filter} → {len(df)} rows)")
        else:
            print(f"  ⚠ Unique filter column '{UNIQUE_FILTER_COL}' not found — skipping filter.")

        # 2e. Reset index & save
        df.reset_index(drop=True, inplace=True)
        self.df = df
        df.to_csv(self.clean_path, index=False)
        print(f"  ✔ Cleaned dataset saved → '{self.clean_path}'  shape: {df.shape}")

    # ------------------------------------------------------------------
    # STAGE 3 — ENGINEERING DATA ANALYTICS (NumPy-based)
    # ------------------------------------------------------------------
    def analyze(self) -> None:
        """
        Compute and store:
          • Descriptive Statistics  (mean, median, std, variance)
          • Distribution Analysis   (skewness, IQR, outliers)
          • Correlation Analysis    (Pearson r)
          • Comparative Analysis    (low vs high cutting speed)
        All numerical computations use NumPy.
        """
        print("\n[STAGE 3] Engineering Data Analytics...")
        df  = self.df
        col = self.target_col
        arr = df[col].dropna().to_numpy(dtype=np.float64)

        # --- Descriptive Statistics ---
        mean   = np.mean(arr)
        median = np.median(arr)
        std    = np.std(arr, ddof=1)
        var    = np.var(arr, ddof=1)

        self.stats["mean"]   = mean
        self.stats["median"] = median
        self.stats["std"]    = std
        self.stats["var"]    = var

        print(f"\n  Descriptive Statistics for '{col}':")
        print(f"    Mean              : {mean:.4f} µm")
        print(f"    Median            : {median:.4f} µm")
        print(f"    Std Deviation     : {std:.4f} µm")
        print(f"    Variance          : {var:.4f} µm²")

        # --- Distribution Analysis ---
        skewness = stats.skew(arr)
        q1, q3   = np.percentile(arr, [25, 75])
        iqr      = q3 - q1
        lower_f  = q1 - 1.5 * iqr
        upper_f  = q3 + 1.5 * iqr
        outliers = arr[(arr < lower_f) | (arr > upper_f)]

        self.stats.update({
            "skewness": skewness,
            "iqr": iqr,
            "outlier_count": len(outliers),
            "lower_fence": lower_f,
            "upper_fence": upper_f,
        })

        print(f"\n  Distribution Analysis:")
        print(f"    Skewness          : {skewness:.4f}  "
              f"({'right-skewed' if skewness > 0 else 'left-skewed' if skewness < 0 else 'symmetric'})")
        print(f"    IQR               : {iqr:.4f} µm")
        print(f"    Outlier Count     : {len(outliers)} "
              f"(fences: [{lower_f:.4f}, {upper_f:.4f}])")

        # --- Correlation Analysis ---
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        self.stats["corr_matrix"] = corr_matrix

        if col in corr_matrix.columns:
            corr_with_target = corr_matrix[col].drop(col).sort_values(key=abs, ascending=False)
            self.stats["corr_with_target"] = corr_with_target
            print(f"\n  Correlation with '{col}' (Pearson r):")
            for feat, r in corr_with_target.items():
                direction = "positive" if r > 0 else "negative"
                strength  = "strong" if abs(r) > 0.6 else "moderate" if abs(r) > 0.3 else "weak"
                print(f"    {feat:<30} r = {r:+.4f}  ({strength} {direction})")

        # --- Comparative Analysis: Low vs High cutting speed ---
        speed_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["vc", "speed", "rpm", "v"])),
            None
        )
        if speed_col:
            median_speed = np.median(df[speed_col].to_numpy(dtype=np.float64))
            low_group    = df[df[speed_col] <= median_speed][col].dropna().to_numpy(np.float64)
            high_group   = df[df[speed_col] >  median_speed][col].dropna().to_numpy(np.float64)

            t_stat, p_val = stats.ttest_ind(low_group, high_group, equal_var=False)
            self.stats.update({
                "speed_col": speed_col,
                "low_mean":  np.mean(low_group),
                "high_mean": np.mean(high_group),
                "t_stat":    t_stat,
                "p_val":     p_val,
            })

            print(f"\n  Comparative Analysis ('{speed_col}' — Low vs High):")
            print(f"    Low-speed  mean Ra : {np.mean(low_group):.4f} µm  (n={len(low_group)})")
            print(f"    High-speed mean Ra : {np.mean(high_group):.4f} µm  (n={len(high_group)})")
            print(f"    Welch t-test       : t = {t_stat:.4f},  p = {p_val:.4f}")
            significance = "statistically significant" if p_val < 0.05 else "not statistically significant"
            print(f"    → Difference is {significance} (α = 0.05).")

        print("\n  ✔ Analysis complete.")

    # ------------------------------------------------------------------
    # STAGE 4A — STATIC VISUALIZATIONS (min 3)
    # ------------------------------------------------------------------
    def visualize_static(self) -> None:
        """Generate and save at least 3 static plots."""
        print("\n[STAGE 4A] Static Visualizations...")
        df  = self.df
        col = self.target_col
        s   = self.stats

        plt.style.use("seaborn-v0_8-whitegrid")
        palette = sns.color_palette("mako", 8)

        # ── Plot 1: Histogram + KDE of Surface Roughness ──────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[col].dropna(), bins=25, color=palette[2], edgecolor="white",
                alpha=0.85, density=True, label="Ra distribution")
        kde_x = np.linspace(df[col].min(), df[col].max(), 300)
        kde   = stats.gaussian_kde(df[col].dropna())
        ax.plot(kde_x, kde(kde_x), color=palette[5], lw=2.5, label="KDE")
        ax.axvline(s["mean"],   color="tomato",   lw=2, ls="--", label=f"Mean={s['mean']:.3f}")
        ax.axvline(s["median"], color="gold",     lw=2, ls=":",  label=f"Median={s['median']:.3f}")
        ax.set_title(f"Distribution of Surface Roughness Ra (µm)\n{STUDENT_NAME} | {STUDENT_ID}")
        ax.set_xlabel("Ra (µm)")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "plot1_histogram.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✔ Saved: {path}")

        # ── Plot 2: Boxplot — Low vs High cutting speed ────────────────
        speed_col = s.get("speed_col")
        if speed_col:
            med_speed = np.median(df[speed_col].to_numpy(np.float64))
            df_box = df[[speed_col, col]].copy()
            df_box["Speed Group"] = np.where(df_box[speed_col] <= med_speed,
                                             "Low Speed", "High Speed")
            fig, ax = plt.subplots(figsize=(7, 5))
            groups    = ["Low Speed", "High Speed"]
            data_list = [df_box[df_box["Speed Group"] == g][col].dropna().values for g in groups]
            bp = ax.boxplot(data_list, patch_artist=True,
                            medianprops=dict(color="gold", lw=2))
            for patch, color in zip(bp["boxes"], [palette[1], palette[4]]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            ax.set_xticklabels(groups)
            ax.set_title(f"Ra Comparison: Low vs High Cutting Speed\n{STUDENT_NAME} | {STUDENT_ID}")
            ax.set_ylabel("Ra (µm)")
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, "plot2_boxplot_speed.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  ✔ Saved: {path}")

        # ── Plot 3: Correlation Heatmap ────────────────────────────────
        corr = s.get("corr_matrix")
        if corr is not None and not corr.empty:
            fig, ax = plt.subplots(figsize=(9, 7))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                        center=0, linewidths=0.5, ax=ax,
                        cbar_kws={"shrink": 0.8})
            ax.set_title(f"Pearson Correlation Heatmap\n{STUDENT_NAME} | {STUDENT_ID}")
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, "plot3_heatmap.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  ✔ Saved: {path}")

        # ── Plot 4: Scatter — Feed Rate vs Ra ─────────────────────────
        feed_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["f", "feed"])),
            None
        )
        if feed_col and feed_col != col:
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(df[feed_col], df[col], c=df[col], cmap="plasma",
                            alpha=0.75, edgecolors="white", linewidths=0.4, s=50)
            plt.colorbar(sc, ax=ax, label="Ra (µm)")
            # linear trend
            mask = df[[feed_col, col]].dropna()
            m, b, *_ = stats.linregress(mask[feed_col], mask[col])
            xline = np.linspace(mask[feed_col].min(), mask[feed_col].max(), 200)
            ax.plot(xline, m * xline + b, "r--", lw=2, label=f"Trend: y={m:.3f}x+{b:.3f}")
            ax.set_xlabel(f"{feed_col} (mm/rev)")
            ax.set_ylabel("Ra (µm)")
            ax.set_title(f"Feed Rate vs Surface Roughness Ra\n{STUDENT_NAME} | {STUDENT_ID}")
            ax.legend()
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, "plot4_scatter_feed_vs_Ra.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  ✔ Saved: {path}")

        # ── Plot 5: Outlier detection — Z-score ───────────────────────
        arr    = df[col].dropna().to_numpy(np.float64)
        z_scores = np.abs(stats.zscore(arr))
        fig, ax  = plt.subplots(figsize=(9, 4))
        idx      = np.arange(len(arr))
        colors   = ["tomato" if z > 3 else "#2d6a8f" for z in z_scores]
        ax.scatter(idx, arr, c=colors, s=20, alpha=0.7)
        ax.axhline(s["lower_fence"], color="orange", ls="--", lw=1.5, label="Lower fence (IQR)")
        ax.axhline(s["upper_fence"], color="orange", ls="--", lw=1.5, label="Upper fence (IQR)")
        ax.set_xlabel("Record Index")
        ax.set_ylabel("Ra (µm)")
        ax.set_title(f"Outlier Detection — Surface Roughness Ra\n{STUDENT_NAME} | {STUDENT_ID}")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "plot5_outlier_detection.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✔ Saved: {path}")

    # ------------------------------------------------------------------
    # STAGE 4B — ANIMATED VISUALIZATIONS (min 2)
    # ------------------------------------------------------------------
    def visualize_animated(self) -> None:
        """Generate and save at least 2 animated plots."""
        print("\n[STAGE 4B] Animated Visualizations...")
        df  = self.df
        col = self.target_col
        palette = sns.color_palette("mako", 8)

        # ── Animation 1: Rolling Mean of Ra over records ──────────────
        arr     = df[col].dropna().reset_index(drop=True).to_numpy(np.float64)
        n       = len(arr)
        win     = max(5, n // 20)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_xlim(0, n)
        ax.set_ylim(arr.min() * 0.9, arr.max() * 1.1)
        ax.set_xlabel("Record Index")
        ax.set_ylabel("Ra (µm)")
        ax.set_title(f"Animated Rolling Mean of Ra\n{STUDENT_NAME} | {STUDENT_ID}")
        raw_line,  = ax.plot([], [], color=palette[1], alpha=0.4, lw=1, label="Ra (raw)")
        roll_line, = ax.plot([], [], color="tomato",   lw=2.5,   label=f"Rolling Mean (w={win})")
        ax.legend(loc="upper right")

        def _init1():
            raw_line.set_data([], [])
            roll_line.set_data([], [])
            return raw_line, roll_line

        def _update1(frame):
            end = frame + 1
            x   = np.arange(end)
            raw_line.set_data(x, arr[:end])
            if end >= win:
                rm = np.convolve(arr[:end], np.ones(win) / win, mode="valid")
                roll_line.set_data(np.arange(win - 1, end), rm)
            return raw_line, roll_line

        frames = np.linspace(win, n - 1, min(80, n - win), dtype=int)
        ani1 = animation.FuncAnimation(fig, _update1, frames=frames,
                                       init_func=_init1, blit=True, interval=80)
        path1 = os.path.join(OUTPUT_DIR, "anim1_rolling_mean_Ra.gif")
        ani1.save(path1, writer="pillow", fps=15)
        plt.close()
        print(f"  ✔ Saved: {path1}")

        # ── Animation 2: Progressive Histogram Build-up ───────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Ra (µm)")
        ax.set_ylabel("Count")
        ax.set_title(f"Progressive Distribution Build-up (Ra)\n{STUDENT_NAME} | {STUDENT_ID}")

        bin_edges = np.histogram_bin_edges(arr, bins=25)

        def _update2(frame):
            ax.cla()
            ax.set_xlabel("Ra (µm)")
            ax.set_ylabel("Count")
            ax.set_title(f"Progressive Distribution Build-up (Ra)\n{STUDENT_NAME} | {STUDENT_ID}")
            slice_ = arr[:frame + 1]
            ax.hist(slice_, bins=bin_edges, color=palette[3], edgecolor="white", alpha=0.85)
            ax.axvline(np.mean(slice_), color="tomato", lw=2, ls="--",
                       label=f"Mean={np.mean(slice_):.3f}")
            ax.legend(loc="upper right")

        step_frames = np.linspace(5, n - 1, min(60, n - 5), dtype=int)
        ani2 = animation.FuncAnimation(fig, _update2, frames=step_frames,
                                       interval=120, repeat=False)
        path2 = os.path.join(OUTPUT_DIR, "anim2_progressive_histogram.gif")
        ani2.save(path2, writer="pillow", fps=12)
        plt.close()
        print(f"  ✔ Saved: {path2}")

    # ------------------------------------------------------------------
    # STAGE 5 — SUMMARY REPORT (console)
    # ------------------------------------------------------------------
    def report(self) -> None:
        """Print a concise engineering summary to the console."""
        s   = self.stats
        col = self.target_col
        print("\n" + "="*65)
        print("  ENGINEERING SUMMARY REPORT")
        print(f"  {TOPIC}")
        print(f"  {STUDENT_NAME}  |  {STUDENT_ID}")
        print("="*65)
        print(f"  Target Variable  : {col} (Surface Roughness Ra, µm)")
        print(f"  Cleaned Records  : {len(self.df)}")
        print(f"\n  ┌─ Descriptive Statistics ──────────────────────────┐")
        print(f"  │  Mean         : {s['mean']:.4f} µm                    │")
        print(f"  │  Median       : {s['median']:.4f} µm                    │")
        print(f"  │  Std Dev      : {s['std']:.4f} µm                    │")
        print(f"  │  Variance     : {s['var']:.4f} µm²                   │")
        print(f"  │  Skewness     : {s['skewness']:.4f}                        │")
        print(f"  │  IQR          : {s['iqr']:.4f} µm                    │")
        print(f"  │  Outliers     : {s['outlier_count']} records                     │")
        print(f"  └───────────────────────────────────────────────────┘")
        if "t_stat" in s:
            sig = "✔ Significant" if s["p_val"] < 0.05 else "✘ Not Significant"
            print(f"\n  Comparative (Low vs High Speed):")
            print(f"    Low-speed Ra  : {s['low_mean']:.4f} µm")
            print(f"    High-speed Ra : {s['high_mean']:.4f} µm")
            print(f"    Welch t-test  : p = {s['p_val']:.4f}  → {sig} (α=0.05)")
        print(f"\n  Output files saved to: '{OUTPUT_DIR}/'")
        print("="*65)
        print("  Pipeline complete. Good luck on your IEEE paper!")
        print("="*65 + "\n")

    # ------------------------------------------------------------------
    # MAIN RUNNER
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute all pipeline stages in sequence."""
        self.ingest()
        self.clean()
        self.analyze()
        self.visualize_static()
        self.visualize_animated()
        self.report()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    pipeline = CNCPipeline(raw_path=RAW_FILE, clean_path=CLEAN_FILE)
    pipeline.run()
