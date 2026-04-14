"""
Funding Quantile Transition Matrices and Mobility Indices:
A Comparative Analysis of China and the United States

This script:
    1. Computes funding-quantile transition matrices for the full sample
       (without disciplinary stratification);
    2. Plots 2x3 transition-matrix heatmaps for China and the United States;
    3. Plots comparative Shorrocks and Bartholomew indices;
    4. Exports publication-quality, high-resolution figures.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# ===========================
# 1. Global style settings
# ===========================
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set(style="whitegrid", font="Arial", font_scale=1.1)
cmap = sns.color_palette("Blues", as_cmap=True)

# Output directories
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# ===========================
# 2. Utility functions
# ===========================
def assign_quantile_group(subdf):
    """
    Assign funding quantile groups within each period.
    """
    q10, q30, q50, q70, q90 = [subdf["Funding"].quantile(x) for x in [0.1, 0.3, 0.5, 0.7, 0.9]]

    def label(f):
        if f >= q90:
            return "Top 10%"
        elif f >= q70:
            return "10-30%"
        elif f >= q50:
            return "30-50%"
        elif f >= q30:
            return "50-70%"
        elif f >= q10:
            return "70-90%"
        else:
            return "Bottom 10%"

    subdf = subdf.copy()
    subdf["Quantile"] = subdf["Funding"].apply(label)
    return subdf


def compute_shorrocks(matrix):
    """
    Compute the Shorrocks mobility index.
    """
    k = matrix.shape[0]
    return (k - np.trace(matrix.values)) / (k - 1)


def compute_bartholomew(matrix):
    """
    Compute the Bartholomew mobility index.
    """
    M = matrix.values
    k = M.shape[0]
    weights = np.flip(np.arange(1, k + 1))
    upward_index = (1 / k) * np.sum([np.dot(M[i], weights) for i in range(k)]) - (k + 1) / 2
    return upward_index


def process_country(file_path, country_name, output_figure_dir):
    """
    Main pipeline for a given country:
        - aggregate funding by PI and 5-year period
        - assign quantile groups
        - construct adjacent-period transition matrices
        - compute Shorrocks and Bartholomew indices
        - export transition-matrix heatmaps
    """
    df = pd.read_excel(file_path)

    # Define 5-year periods
    df["Period"] = ((df["Year"] - 1990) // 5) * 5 + 1990

    # Aggregate total funding by PI and period
    grouped = df.groupby(["PI", "Period"])["Funding"].sum().reset_index()

    # Assign quantile groups within each period
    labeled = grouped.groupby("Period", group_keys=False).apply(assign_quantile_group)
    labeled.to_excel(os.path.join("tables", f"quantile_groups_{country_name}.xlsx"), index=False)

    # Match adjacent periods
    labeled = labeled.sort_values(["PI", "Period"])
    labeled["Next_Quantile"] = labeled.groupby("PI")["Quantile"].shift(-1)
    labeled["Next_Period"] = labeled.groupby("PI")["Period"].shift(-1)

    valid_transitions = labeled[labeled["Next_Period"] - labeled["Period"] == 5]
    valid_transitions.to_excel(
        os.path.join("tables", f"adjacent_period_transitions_{country_name}.xlsx"),
        index=False
    )

    # Construct transition matrices
    transition_matrices = {}
    shorrocks_indices = {}
    bartholomew_indices = {}

    quantile_order = ["Top 10%", "10-30%", "30-50%", "50-70%", "70-90%", "Bottom 10%"]

    for (period, next_period), group in valid_transitions.groupby(["Period", "Next_Period"]):
        matrix = pd.crosstab(group["Quantile"], group["Next_Quantile"], normalize="index")
        matrix = matrix.reindex(index=quantile_order, columns=quantile_order, fill_value=0)

        transition_matrices[(period, next_period)] = matrix
        shorrocks_indices[(period, next_period)] = compute_shorrocks(matrix)
        bartholomew_indices[(period, next_period)] = compute_bartholomew(matrix)

    print(f"Transition matrices for {country_name}:")
    print(transition_matrices)

    # ===========================
    # 3. Plot transition matrices
    # ===========================
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    sorted_keys = sorted(transition_matrices.keys())

    for i, key in enumerate(sorted_keys[:6]):  # plot first six period pairs
        matrix = transition_matrices[key]
        period, next_period = key
        ax = axes[i]

        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            annot_kws={"size": 8},
            square=True
        )

        ax.set_title(f"{int(period)} to {int(next_period)}", fontsize=12, weight="bold", pad=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("gray")
            spine.set_linewidth(1)

    # Remove unused subplots if necessary
    for j in range(i + 1, 6):
        fig.delaxes(axes[j])

    # Global axis labels
    fig.text(0.5, 0.04, "To Quantile", ha="center", fontsize=14)
    fig.text(0.06, 0.5, "From Quantile", va="center", rotation="vertical", fontsize=14)

    # Figure title
    if country_name == "United States":
        plt.suptitle("United States: Funding Quantile Transition Matrices", fontsize=16, weight="bold")
    else:
        plt.suptitle(f"{country_name}: Funding Quantile Transition Matrices", fontsize=16, weight="bold")

    plt.tight_layout(rect=[0.08, 0.03, 1, 0.97])

    # Save figure
    output_path = os.path.join(output_figure_dir, f"transition_matrix_heatmap_{country_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Combined transition heatmap saved to: {output_path}")

    return shorrocks_indices, bartholomew_indices


# ===========================
# 4. Process China and the United States
# ===========================
china_shorrocks, china_bartholomew = process_country(
    "China_NSFC_data.xlsx", "China", "figures"
)

usa_shorrocks, usa_bartholomew = process_country(
    "US_funding_data.xlsx", "United States", "figures"
)

# ===========================
# 5. Plot Shorrocks and Bartholomew indices
# ===========================
period_labels_china = [f"{int(p[0])}-{int(p[1])}" for p in china_shorrocks.keys()]
period_labels_usa = [f"{int(p[0])}-{int(p[1])}" for p in usa_shorrocks.keys()]

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "figure.dpi": 400,
    "axes.linewidth": 0.8
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# ----- (a) Shorrocks index -----
china_values = list(china_shorrocks.values())
usa_values = list(usa_shorrocks.values())
china_mean = np.mean(china_values)
usa_mean = np.mean(usa_values)

axes[0].plot(
    period_labels_china, china_values,
    color="grey", marker="o", linewidth=2.0,
    label=f"China (mean={china_mean:.2f})"
)
axes[0].fill_between(period_labels_china, china_values, alpha=0.25, color="grey")

axes[0].plot(
    period_labels_usa, usa_values,
    color="#7F91B9", marker="s", linewidth=2.0,
    label=f"United States (mean={usa_mean:.2f})"
)
axes[0].fill_between(period_labels_usa, usa_values, alpha=0.25, color="#7F91B9")

axes[0].set_xlabel("Period", labelpad=4)
axes[0].set_ylabel("Index Value", labelpad=6)
axes[0].set_ylim(0, 1.2)
axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
axes[0].legend(frameon=False, loc="upper left")

for spine in axes[0].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color("black")

axes[0].text(
    0.5, 1.08, "(a) Shorrocks Index (Mobility)",
    transform=axes[0].transAxes,
    fontsize=12, fontweight="bold",
    va="bottom", ha="center"
)

# ----- (b) Bartholomew index -----
china_values_b = list(china_bartholomew.values())
usa_values_b = list(usa_bartholomew.values())
china_mean_b = np.mean(china_values_b)
usa_mean_b = np.mean(usa_values_b)

axes[1].plot(
    period_labels_china, china_values_b,
    color="grey", marker="o", linewidth=2.0,
    label=f"China (mean={china_mean_b:.2f})"
)
axes[1].fill_between(period_labels_china, china_values_b, alpha=0.25, color="grey")

axes[1].plot(
    period_labels_usa, usa_values_b,
    color="#7F91B9", marker="s", linewidth=2.0,
    label=f"United States (mean={usa_mean_b:.2f})"
)
axes[1].fill_between(period_labels_usa, usa_values_b, alpha=0.25, color="#7F91B9")

axes[1].set_xlabel("Period", labelpad=4)
axes[1].set_ylabel("Index Value", labelpad=6)
axes[1].set_ylim(axes[0].get_ylim())
axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
axes[1].legend(frameon=False, loc="upper left")

for spine in axes[1].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color("black")

axes[1].text(
    0.5, 1.08, "(b) Bartholomew Index (Upward Mobility)",
    transform=axes[1].transAxes,
    fontsize=12, fontweight="bold",
    va="bottom", ha="center"
)

for ax in axes:
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

plt.tight_layout(w_pad=2.5, h_pad=1.0)

# Save figure
comparison_path = os.path.join("figures", "Figure6_Mobility_Indices_China_US.png")
plt.savefig(comparison_path, dpi=600, bbox_inches="tight")
plt.close()

print(f"Publication-quality figure saved to: {comparison_path}")