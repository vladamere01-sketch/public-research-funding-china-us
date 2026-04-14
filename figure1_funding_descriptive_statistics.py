import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ===========================
# 1. Load datasets
# ===========================
us_path = "US_funding_data.xlsx"
cn_path = "China_NSFC_data.xlsx"

us = pd.read_excel(us_path)
cn = pd.read_excel(cn_path)

# ===========================
# 2. Data preprocessing (harmonized time window: 1990–2020)
# ===========================
def preprocess(df, funding_col, count_col):
    df['Year'] = df['Year'].astype(int)
    df = df[(df['Year'] >= 1990) & (df['Year'] <= 2020)]
    yearly = df.groupby('Year').agg(
        Total_Funding=(funding_col, 'sum'),
        Project_Count=(count_col, 'count')
    ).reset_index()
    return yearly

us_stats = preprocess(us, 'Funding', 'AwardNumber')
cn_stats = preprocess(cn, 'Funding', 'project_num')

# Unit conversion
us_stats['Funding_Unit'] = us_stats['Total_Funding'] / 1e6   # Million USD
cn_stats['Funding_Unit'] = cn_stats['Total_Funding'] / 1e2   # Million RMB (10,000 CNY units)

# ===========================
# 3. Global plotting style
# ===========================
plt.style.use('default')
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ===========================
# 4. Create two-panel figure
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=400)

# Colors and transparency
bar_color = 'gray'
line_color = '#293890'
bar_alpha = 0.4

# ===========================
# Panel (a): United States
# ===========================
ax1 = axes[0]
ax1_line = ax1
ax1_bar = ax1.twinx()

ax1_line.plot(us_stats['Year'], us_stats['Funding_Unit'],
              color=line_color, marker='o', linewidth=2.2,
              label='Total Funding (Million USD)')

ax1_bar.bar(us_stats['Year'], us_stats['Project_Count'],
            color=bar_color, alpha=bar_alpha, width=0.8,
            label='Number of Projects')

ax1_line.set_xlabel('Year', fontsize=12)
ax1_line.set_ylabel('Total Funding (Million USD)', fontsize=12, color=line_color)
ax1_bar.set_ylabel('Number of Projects', fontsize=12, color=bar_color)
ax1_bar.set_ylim(4000, 10000)

ax1_line.tick_params(axis='y', labelcolor=line_color)
ax1_bar.tick_params(axis='y', labelcolor=bar_color)

ax1_line.set_xlim(1989.5, 2020.5)
ax1_line.set_xticks(range(1990, 2021, 5))
ax1_line.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))

# Legend
lines_1, labels_1 = ax1_line.get_legend_handles_labels()
lines_2, labels_2 = ax1_bar.get_legend_handles_labels()
ax1_line.legend(lines_1 + lines_2, labels_1 + labels_2,
                loc='upper left', frameon=False, fontsize=10)

ax1_line.set_title('(a) United States (1990–2020)', fontsize=13, fontweight='bold')

# Borders
for spine in ax1_line.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
for spine in ax1_bar.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

# ===========================
# Panel (b): China
# ===========================
ax2 = axes[1]
ax2_line = ax2
ax2_bar = ax2.twinx()

ax2_line.plot(cn_stats['Year'], cn_stats['Funding_Unit'],
              color=line_color, marker='o', linewidth=2.2,
              label='Total Funding (Million RMB)')

ax2_bar.bar(cn_stats['Year'], cn_stats['Project_Count'],
            color=bar_color, alpha=bar_alpha, width=0.8,
            label='Number of Projects')

ax2_line.set_xlabel('Year', fontsize=12)
ax2_line.set_ylabel('Total Funding (Million RMB)', fontsize=12, color=line_color)
ax2_bar.set_ylabel('Number of Projects', fontsize=12, color=bar_color)

ax2_line.tick_params(axis='y', labelcolor=line_color)
ax2_bar.tick_params(axis='y', labelcolor=bar_color)

ax2_line.set_xlim(1989.5, 2020.5)
ax2_line.set_xticks(range(1990, 2021, 5))
ax2_line.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))

# Legend
lines_1, labels_1 = ax2_line.get_legend_handles_labels()
lines_2, labels_2 = ax2_bar.get_legend_handles_labels()
ax2_line.legend(lines_1 + lines_2, labels_1 + labels_2,
                loc='upper left', frameon=False, fontsize=10)

ax2_line.set_title('(b) China (1990–2020)', fontsize=13, fontweight='bold')

# Borders
for spine in ax2_line.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
for spine in ax2_bar.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

# ===========================
# 5. Layout and export
# ===========================
plt.tight_layout(w_pad=3)

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "Figure1_Funding_and_Projects_US_China.png")

plt.savefig(output_path, dpi=400, bbox_inches='tight')
plt.show()

print(f"Figure saved to: {output_path}")