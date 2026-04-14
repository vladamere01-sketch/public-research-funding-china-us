import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import os
from matplotlib.gridspec import GridSpec

# ======================
# 1. Global style settings
# ======================
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set(style="whitegrid", font="Arial", font_scale=1.0)

# ======================
# 2. Function to compute Top-k funding shares
# ======================
def compute_topk_shares(df, k_list):
    results = []

    for year, group in df.groupby("Year"):
        row = {"Year": year}
        group_sorted = group.sort_values(by="Funding", ascending=False)
        total_funding = group_sorted["Funding"].sum()

        for k in k_list:
            top_n = max(1, int(len(group_sorted) * k))
            share = (
                group_sorted["Funding"].head(top_n).sum() / total_funding
                if total_funding > 0
                else np.nan
            )
            row[f"Top {int(k * 100)}%"] = share

        results.append(row)

    return pd.DataFrame(results).sort_values("Year")


# ======================
# 3. Load and process data
# ======================
china_df = pd.read_excel("China_NSFC_data.xlsx")
us_df = pd.read_excel("US_funding_data.xlsx")

k_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

china_share = compute_topk_shares(china_df, k_list).set_index("Year") * 100
us_share = compute_topk_shares(us_df, k_list).set_index("Year") * 100

china_mean = china_share.mean()
china_std = china_share.std()
us_mean = us_share.mean()
us_std = us_share.std()

# ======================
# 4. Create figure layout (one row, three columns)
# ======================
fig = plt.figure(figsize=(12, 3))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

# Panel allocation
ax1 = fig.add_subplot(gs[0, 1])  # (b) United States heatmap
ax2 = fig.add_subplot(gs[0, 2])  # (c) China heatmap
ax3 = fig.add_subplot(gs[0, 0])  # (a) line chart

# ======================
# 5. Draw heatmaps with shared colorbar
# ======================
vmin = min(us_share.min().min(), china_share.min().min())
vmax = max(us_share.max().max(), china_share.max().max())

# United States heatmap
hm1 = sns.heatmap(
    us_share.T,
    ax=ax1,
    cmap="coolwarm",
    vmin=vmin,
    vmax=vmax,
    annot=False,
    cbar=False,
    linewidths=0.1,
    linecolor="gray",
)
ax1.set_title("(b) United States: Top-k Funding Share", fontsize=12, weight="bold")
ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Top-k Group", fontsize=11)
ax1.tick_params(axis="x", rotation=45, labelsize=9)
ax1.tick_params(axis="y", labelsize=9)

# China heatmap
hm2 = sns.heatmap(
    china_share.T,
    ax=ax2,
    cmap="coolwarm",
    vmin=vmin,
    vmax=vmax,
    annot=False,
    cbar=False,
    linewidths=0.1,
    linecolor="gray",
)
ax2.set_title("(c) China: Top-k Funding Share", fontsize=12, weight="bold")
ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("")
ax2.tick_params(axis="x", rotation=45, labelsize=9)
ax2.tick_params(axis="y", labelsize=9)

# Shared colorbar
cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
cbar = fig.colorbar(hm2.collections[0], cax=cbar_ax)
cbar.set_label("Funding Share (%)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# ======================
# 6. Draw line chart
# ======================
topk_labels = [f"Top {int(k * 100)}%" for k in k_list]

print("China mean:")
print(china_mean)
print("United States mean:")
print(us_mean)

ax3.plot(
    topk_labels,
    china_mean,
    label="China",
    color="tomato",
    marker="o",
    linewidth=2,
)
ax3.fill_between(
    topk_labels,
    china_mean - china_std,
    china_mean + china_std,
    color="tomato",
    alpha=0.25,
)

ax3.plot(
    topk_labels,
    us_mean,
    label="United States",
    color="royalblue",
    marker="s",
    linewidth=2,
)
ax3.fill_between(
    topk_labels,
    us_mean - us_std,
    us_mean + us_std,
    color="royalblue",
    alpha=0.25,
)

ax3.set_title("(a) Average Top-k Funding Share", fontsize=12, weight="bold")
ax3.set_xlabel("Top-k Group", fontsize=11)
ax3.set_ylabel("Funding Share (%)", fontsize=11)
ax3.tick_params(axis="x", rotation=45, labelsize=9)
ax3.tick_params(axis="y", labelsize=9)
ax3.legend(fontsize=10, loc="upper left", frameon=False)
ax3.grid(True, linestyle="--", alpha=0.4)

# ======================
# 7. Save figure
# ======================
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "Figure3_Topk_Funding_Shares_US_China.png")
plt.savefig(output_path, dpi=400, bbox_inches="tight")
plt.show()

print(f"Figure saved to: {output_path}")