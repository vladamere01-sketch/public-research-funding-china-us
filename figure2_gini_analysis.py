import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===========================
# 1. Gini coefficient function
# ===========================
def gini_coefficient(x):
    x = np.array(x, dtype=float)
    if np.all(x == 0):
        return 0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


# ===========================
# 2. Gini calculation by period (aggregation approach)
# ===========================
def compute_gini_by_period(df, country_name, period, corrected_ratio=0.7):
    results = []

    df = df.copy()
    BASE_YEAR = 1990

    df["Period_Start"] = ((df["Year"] - BASE_YEAR) // period) * period + BASE_YEAR
    df["Period_Label"] = (
        df["Period_Start"].astype(str)
        + "-"
        + (df["Period_Start"] + period - 1).astype(str)
    )

    for label, group in df.groupby("Period_Label"):

        # --- Individual level ---
        indiv = group.groupby(["PI", "Organization"])["Funding"].sum()
        indiv_gini = gini_coefficient(indiv)

        # Correction: include unfunded researchers (zeros)
        n_extra = int(len(indiv) * corrected_ratio)
        indiv_corr = np.concatenate([indiv.values, np.zeros(n_extra)])
        indiv_corr_gini = gini_coefficient(indiv_corr)

        # --- Institutional level ---
        org = group.groupby("Organization")["Funding"].sum()
        org_gini = gini_coefficient(org)

        # --- Regional level ---
        region = group.groupby("State")["Funding"].sum()
        region_gini = gini_coefficient(region)

        results.append({
            "Country": country_name,
            "Period": label,
            "Individual": indiv_gini,
            "Individual_Corrected": indiv_corr_gini,
            "Institution": org_gini,
            "Region": region_gini
        })

    return pd.DataFrame(results)


# ===========================
# 3. Load and preprocess data
# ===========================
us_df = pd.read_excel("US_funding_data.xlsx")
cn_df = pd.read_excel("China_NSFC_data.xlsx")

for df in [us_df, cn_df]:
    df["Year"] = df["Year"].astype(int)
    df.query("1990 <= Year <= 2020", inplace=True)

# 1-year window (baseline correction = 70%)
us_1y = compute_gini_by_period(us_df, "United States", 1, corrected_ratio=0.7)
cn_1y = compute_gini_by_period(cn_df, "China", 1, corrected_ratio=0.7)

# 3-year window (reduced correction)
us_3y = compute_gini_by_period(us_df, "United States", 3, corrected_ratio=0.6)
cn_3y = compute_gini_by_period(cn_df, "China", 3, corrected_ratio=0.6)

# 5-year window (further reduced correction)
us_5y = compute_gini_by_period(us_df, "United States", 5, corrected_ratio=0.5)
cn_5y = compute_gini_by_period(cn_df, "China", 5, corrected_ratio=0.5)


# ===========================
# 4. Plot style
# ===========================
colors = {
    "Individual": "#293890",
    "Individual_Corrected": "#293890",
    "Institution": "grey",
    "Region": "#E7A494"
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
})

# ===========================
# 5. Plotting function
# ===========================
def plot_panel(ax, x, df, title, show_corrected):
    ax.plot(x, df["Individual"], lw=2,
            label=f"Individual (Mean={df['Individual'].mean():.3f})",
            color=colors["Individual"])

    if show_corrected:
        ax.plot(x, df["Individual_Corrected"], lw=2, ls="--",
                label=f"Individual (Corrected, Mean={df['Individual_Corrected'].mean():.3f})",
                color=colors["Individual"])

    ax.plot(x, df["Institution"], lw=2,
            label=f"Institution (Mean={df['Institution'].mean():.3f})",
            color=colors["Institution"])

    ax.plot(x, df["Region"], lw=2,
            label=f"Region (Mean={df['Region'].mean():.3f})",
            color=colors["Region"])

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)


# ===========================
# 6. Multi-panel figure (3×2)
# ===========================
fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=400, sharey=True)

# --- 1-year ---
x1 = us_1y["Period"].str.split("-").str[0].astype(int)
plot_panel(axes[0, 0], x1, us_1y, "(a) United States (1-Year)", True)
plot_panel(axes[1, 0], x1, cn_1y, "(d) China (1-Year)", True)

axes[0, 0].set_xticks(range(1990, 2021, 5))
axes[1, 0].set_xticks(range(1990, 2021, 5))

# --- 3-year ---
x3 = us_3y["Period"].str.split("-").str[0].astype(int)
plot_panel(axes[0, 1], x3, us_3y, "(b) United States (3-Year)", False)
plot_panel(axes[1, 1], x3, cn_3y, "(e) China (3-Year)", False)

axes[0, 1].set_xticks(range(1990, 2021, 3))
axes[1, 1].set_xticks(range(1990, 2021, 3))

# --- 5-year ---
x5 = us_5y["Period"].str.split("-").str[0].astype(int)
plot_panel(axes[0, 2], x5, us_5y, "(c) United States (5-Year)", False)
plot_panel(axes[1, 2], x5, cn_5y, "(f) China (5-Year)", False)

axes[0, 2].set_xticks(range(1990, 2021, 5))
axes[1, 2].set_xticks(range(1990, 2021, 5))

# Axis labels
axes[0, 0].set_ylabel("Gini Coefficient")
axes[1, 0].set_ylabel("Gini Coefficient")

axes[1, 0].set_xlabel("Year")
axes[1, 1].set_xlabel("Period")
axes[1, 2].set_xlabel("Period")

plt.tight_layout()

# ===========================
# 7. Save figure
# ===========================
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "Figure2_Gini_Analysis_US_China.png")

plt.savefig(output_path, bbox_inches="tight", dpi=400)
plt.show()

print(f"Figure saved to: {output_path}")