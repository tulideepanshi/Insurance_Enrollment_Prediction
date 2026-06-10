"""
Exploratory Data Analysis — Insurance Enrollment Dataset.

Generates all EDA visualizations saved to notebooks/figures/.
Run with: python -m notebooks.eda

Includes enrollment rate curves by numeric features showing
the step-function behavior in age, monotonic salary relationship,
and flat tenure effect.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_DATA_PATH, TARGET_COL

# ── Setup ──
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

df = pd.read_csv(RAW_DATA_PATH)
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target distribution: {df[TARGET_COL].value_counts().to_dict()}")


# ── 1. Target distribution ──
fig, ax = plt.subplots(figsize=(6, 4))
df[TARGET_COL].value_counts().plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
ax.set_title("Target Distribution (enrolled)")
ax.set_xticklabels(["Not Enrolled (0)", "Enrolled (1)"], rotation=0)
ax.set_ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "target_distribution.png"), dpi=150)
plt.close()


# ── 2. Categorical enrollment rates ──
fig, axes = plt.subplots(1, len(CATEGORICAL_FEATURES), figsize=(20, 5))
for ax, col in zip(axes, CATEGORICAL_FEATURES):
    rates = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
    rates.plot(kind="bar", ax=ax, color="#3498db")
    ax.set_title(f"Enrollment Rate by {col}")
    ax.set_ylabel("Enrollment Rate")
    ax.set_ylim(0, 1)
    ax.axhline(y=df[TARGET_COL].mean(), color="red", linestyle="--", alpha=0.7, label="Overall")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "categorical_enrollment_rates.png"), dpi=150)
plt.close()


# ── 3. Association heatmap (all features) ──
# Uses the statistically correct metric for each feature pair:
#   numeric vs numeric:       Pearson correlation
#   categorical vs categorical: Cramér's V (chi-squared based)
#   numeric vs categorical:   Correlation ratio (eta)
# All values are in [0, 1] (strength of association, unsigned).

from scipy.stats import chi2_contingency


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V — association between two categorical variables.

    Based on the chi-squared statistic, corrected for sample size and
    minimum dimension. Returns 0 (no association) to 1 (perfect).
    """
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = len(x)
    r, k = ct.shape
    # Bias-corrected Cramér's V
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    k_corr = k - (k - 1) ** 2 / (n - 1)
    r_corr = r - (r - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return np.sqrt(phi2 / denom) if denom > 0 else 0.0


def _correlation_ratio(categorical: pd.Series, numeric: pd.Series) -> float:
    """Correlation ratio (eta) — association between a categorical and numeric variable.

    Measures the fraction of numeric variance explained by the categorical
    grouping. Returns 0 (no association) to 1 (perfect).
    """
    groups = categorical.astype(str)
    grand_mean = numeric.mean()
    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for _, g in numeric.groupby(groups)
    )
    ss_total = ((numeric - grand_mean) ** 2).sum()
    return np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0


all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
n = len(all_features)
assoc_matrix = np.zeros((n, n))

cat_set = set(CATEGORICAL_FEATURES)
# TARGET_COL is binary — treat as categorical for Cramér's V,
# but also works as numeric for correlation ratio
for i, fi in enumerate(all_features):
    for j, fj in enumerate(all_features):
        if i == j:
            assoc_matrix[i, j] = 1.0
        elif j < i:
            assoc_matrix[i, j] = assoc_matrix[j, i]  # Symmetric
        else:
            fi_cat = fi in cat_set or fi == TARGET_COL
            fj_cat = fj in cat_set or fj == TARGET_COL
            if not fi_cat and not fj_cat:
                # Both numeric → Pearson |r|
                assoc_matrix[i, j] = abs(df[fi].corr(df[fj]))
            elif fi_cat and fj_cat:
                # Both categorical → Cramér's V
                assoc_matrix[i, j] = _cramers_v(df[fi], df[fj])
            elif fi_cat:
                # fi categorical, fj numeric → correlation ratio
                assoc_matrix[i, j] = _correlation_ratio(df[fi], df[fj])
            else:
                # fi numeric, fj categorical → correlation ratio
                assoc_matrix[i, j] = _correlation_ratio(df[fj], df[fi])

assoc_df = pd.DataFrame(assoc_matrix, index=all_features, columns=all_features)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(assoc_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8, "label": "Association strength"})
ax.set_title("Feature Association Matrix\n(Pearson | Cramér's V | Correlation Ratio)",
             fontsize=13, pad=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()


# ── 4. Age vs salary scatter by enrollment ──
fig, ax = plt.subplots(figsize=(8, 6))
for enrolled, group in df.groupby(TARGET_COL):
    label = "Enrolled" if enrolled == 1 else "Not Enrolled"
    ax.scatter(group["age"], group["salary"], alpha=0.3, s=10, label=label)
ax.set_xlabel("Age")
ax.set_ylabel("Salary")
ax.set_title("Age vs Salary by Enrollment Status")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "age_salary_scatter.png"), dpi=150)
plt.close()


# ── 5. Box plots for outlier visualization ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, NUMERIC_FEATURES):
    df.boxplot(column=col, by=TARGET_COL, ax=ax)
    ax.set_title(f"{col}")
    ax.set_xlabel("Enrolled")
plt.suptitle("Numeric Features by Enrollment", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "boxplots.png"), dpi=150)
plt.close()


# ── 6. Enrollment rate curves by numeric features ──
# This is the key diagnostic showing:
# - Age: step function at 30 (26% → 70%), then plateau
# - Salary: monotonic positive relationship
# - Tenure: flat (near-zero signal)
fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(18, 5))

for ax, col in zip(axes, NUMERIC_FEATURES):
    sorted_df = df.sort_values(col)
    window = min(200, len(df) // 5)
    sorted_df["rolling_rate"] = sorted_df[TARGET_COL].rolling(window, center=True).mean()

    ax.plot(sorted_df[col], sorted_df["rolling_rate"], color="#3b82f6", linewidth=2, label="Enrollment rate")
    ax.axhline(y=df[TARGET_COL].mean(), color="red", linestyle="--", alpha=0.5, label="Overall rate")
    ax.set_xlabel(col)
    ax.set_ylabel("Enrollment Rate")
    ax.set_title(f"Enrollment Rate by {col}")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("Enrollment Rate Curves — Numeric Features (rolling window)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "enrollment_rate_curves.png"), dpi=150, bbox_inches="tight")
plt.close()


# ── 7. Enrollment rate by binned age (showing step function) ──
fig, ax = plt.subplots(figsize=(8, 5))
age_bins = pd.cut(df["age"], bins=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
enrollment_by_age = df.groupby(age_bins, observed=True)[TARGET_COL].agg(["mean", "count"])

bars = ax.bar(range(len(enrollment_by_age)), enrollment_by_age["mean"], color="#3b82f6", alpha=0.8)
ax.set_xticks(range(len(enrollment_by_age)))
ax.set_xticklabels([str(b) for b in enrollment_by_age.index], rotation=45, ha="right")
ax.set_ylabel("Enrollment Rate")
ax.set_xlabel("Age Bin")
ax.set_title("Enrollment Rate by Age Bin — Step Function at Age 30")
ax.set_ylim(0, 1)
ax.axhline(y=df[TARGET_COL].mean(), color="red", linestyle="--", alpha=0.5, label="Overall rate")

# Annotate the step — position outside bars to avoid overlap
ax.annotate("~26% enrollment\n(under 30)",
            xy=(1, 0.27), xytext=(0.3, 0.50),
            fontsize=9, color="#e74c3c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))
ax.annotate("~70% enrollment\n(30 and above)",
            xy=(3, 0.70), xytext=(5.5, 0.45),
            fontsize=9, color="#2ecc71", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.2))
ax.legend(loc="upper left")

# Add count labels above bars with enough padding
for bar, (_, row) in zip(bars, enrollment_by_age.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f"n={int(row['count'])}", ha="center", fontsize=7, color="gray")

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "age_step_function.png"), dpi=150, bbox_inches="tight")
plt.close()


# ── 8. Enrollment rate by salary decile ──
fig, ax = plt.subplots(figsize=(8, 5))
df["salary_decile"] = pd.qcut(df["salary"], q=10, duplicates="drop")
enrollment_by_salary = df.groupby("salary_decile", observed=True)[TARGET_COL].agg(["mean", "count"])

bars = ax.bar(range(len(enrollment_by_salary)), enrollment_by_salary["mean"], color="#f59e0b", alpha=0.8)
ax.set_xticks(range(len(enrollment_by_salary)))
ax.set_xticklabels([f"${int(b.left/1000)}k-${int(b.right/1000)}k" for b in enrollment_by_salary.index],
                    rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Enrollment Rate")
ax.set_xlabel("Salary Decile")
ax.set_title("Enrollment Rate by Salary Decile")
ax.set_ylim(0, 1)
ax.axhline(y=df[TARGET_COL].mean(), color="red", linestyle="--", alpha=0.5, label="Overall rate")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "salary_enrollment_rate.png"), dpi=150, bbox_inches="tight")
plt.close()

df.drop(columns=["salary_decile"], inplace=True)


print(f"\nAll figures saved to {FIGURE_DIR}/")
print("EDA complete.")
