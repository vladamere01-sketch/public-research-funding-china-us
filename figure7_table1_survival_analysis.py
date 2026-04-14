import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import os

# ============================
# 1. Global plot settings
# ============================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1,
})

# Create output folders
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# ============================
# 2. General function: survival analysis and descriptive statistics
# ============================
def analyze_survival(df_raw, cycle=1):
    """
    Perform Kaplan-Meier survival analysis for Top 10% funding status
    and compute descriptive statistics.
    """
    df = df_raw.copy()
    df["Period"] = ((df["Year"] - df["Year"].min()) // cycle) + 1
    df = df.groupby(["PI_ID", "Period"], as_index=False)["Funding"].sum()

    # --- Identify Top 10% in each period ---
    top_p = 0.1
    df["Top10"] = 0
    for period, group in df.groupby("Period"):
        threshold = group["Funding"].quantile(1 - top_p)
        df.loc[group.index, "Top10"] = (group["Funding"] >= threshold).astype(int)

    # --- Construct spells of continuous Top 10% status ---
    spell_records = []
    for pi, group in df.groupby("PI_ID"):
        group = group.sort_values("Period")
        in_spell = False
        spell_start, spell_len = None, 0

        for _, row in group.iterrows():
            if row["Top10"] == 1:
                if not in_spell:
                    in_spell, spell_start, spell_len = True, row["Period"], 1
                else:
                    spell_len += 1
            else:
                if in_spell:
                    spell_records.append([pi, spell_start, spell_len])
                    in_spell, spell_len = False, 0

        if in_spell:
            spell_records.append([pi, spell_start, spell_len])

    spell_df = pd.DataFrame(spell_records, columns=["PI_ID", "start_period", "spell_length"])

    # --- Prepare survival-analysis dataset ---
    survival_data = []
    for pi, group in df.groupby("PI_ID"):
        group = group.sort_values("Period")
        if group["Top10"].sum() == 0:
            continue

        first_top = group.loc[group["Top10"] == 1, "Period"].min()
        sub = group[group["Period"] >= first_top].reset_index(drop=True)

        T, E = 0, 0
        for _, row in sub.iterrows():
            if row["Top10"] == 1:
                T += 1
            else:
                E = 1
                break

        survival_data.append([pi, T, E])

    survival_df = pd.DataFrame(survival_data, columns=["PI_ID", "T", "E"])

    # --- Merge PI metadata ---
    meta = df_raw[["PI_ID", "PI", "Organization"]].drop_duplicates()
    survival_df = survival_df.merge(meta, on="PI_ID", how="left")
    spell_df = spell_df.merge(meta, on="PI_ID", how="left")

    # ============================
    # 3. Descriptive statistics
    # ============================
    total_pis = df["PI_ID"].nunique()
    active_pis = spell_df["PI_ID"].nunique() if not spell_df.empty else 0

    avg_entries = spell_df.groupby("PI_ID").size().mean() if not spell_df.empty else 0
    avg_cum_time = spell_df.groupby("PI_ID")["spell_length"].sum().mean() if not spell_df.empty else 0
    avg_cont_duration = spell_df["spell_length"].mean() if not spell_df.empty else 0

    # Re-entry intervals
    gaps = []
    for pi, group in spell_df.groupby("PI_ID"):
        if len(group) > 1:
            group = group.sort_values("start_period")
            starts = group["start_period"].values
            lengths = group["spell_length"].values
            ends = starts + lengths - 1
            for i in range(len(starts) - 1):
                gaps.append(starts[i + 1] - ends[i] - 1)

    avg_gap = np.mean(gaps) if len(gaps) > 0 else np.nan

    # Consecutive-cycle statistics
    if not spell_df.empty:
        pi_max_spell = spell_df.groupby("PI_ID")["spell_length"].max().reset_index()
    else:
        pi_max_spell = pd.DataFrame(columns=["PI_ID", "spell_length"])

    consecutive_stats = {}
    for k in range(1, 7):
        num = (pi_max_spell["spell_length"] >= k).sum()
        pct = num / total_pis if total_pis > 0 else np.nan
        consecutive_stats[k] = (num, pct)

    # Maximum continuous duration
    if not pi_max_spell.empty:
        max_spell = pi_max_spell["spell_length"].max()
        num_holders = (pi_max_spell["spell_length"] == max_spell).sum()
    else:
        max_spell = 0
        num_holders = 0

    stats = {
        "total_pis": total_pis,
        "active_pis": active_pis,
        "active_pct": active_pis / total_pis if total_pis > 0 else np.nan,
        "avg_entries": avg_entries,
        "avg_cum_time": avg_cum_time,
        "avg_cont_duration": avg_cont_duration,
        "avg_gap": avg_gap,
        "consecutive_stats": consecutive_stats,
        "max_spell": max_spell,
        "num_holders": num_holders
    }

    return survival_df, stats


# ============================
# 4. Helper functions for table formatting
# ============================
def format_count_pct(count, pct):
    return f"{int(count)} ({pct * 100:.2f}%)"

def format_float(x):
    if pd.isna(x):
        return ""
    return f"{x:.2f}"

def format_max_spell(max_spell, num_holders):
    return f"{int(max_spell)} ({int(num_holders)} PIs)"


def build_descriptive_table(china_stats_dict, us_stats_dict):
    """
    Build Table 1 in the requested journal-style format.
    """
    indicators = [
        "PIs entering Top 10%",
        "Avg. number of entries",
        "Avg. cumulative time in Top 10% (cycles)",
        "Avg. Continuous Duration (cycles)",
        "Avg. re-entry interval (cycles)",
        "PIs with ≥ 1 consecutive cycles",
        "PIs with ≥ 2 consecutive cycles",
        "PIs with ≥ 3 consecutive cycles",
        "PIs with ≥ 4 consecutive cycles",
        "PIs with ≥ 5 consecutive cycles",
        "PIs with ≥ 6 consecutive cycles",
        "Max Continuous Duration"
    ]

    rows = []

    country_map = {
        "U.S.": us_stats_dict,
        "China": china_stats_dict
    }

    for indicator in indicators:
        first_row = True
        for country, stats_dict in country_map.items():
            row = {
                "Indicator": indicator if first_row else "",
                "Country": country
            }

            for cycle in [1, 3, 5]:
                s = stats_dict[cycle]

                if indicator == "PIs entering Top 10%":
                    value = format_count_pct(s["active_pis"], s["active_pct"])

                elif indicator == "Avg. number of entries":
                    value = format_float(s["avg_entries"])

                elif indicator == "Avg. cumulative time in Top 10% (cycles)":
                    value = format_float(s["avg_cum_time"])

                elif indicator == "Avg. Continuous Duration (cycles)":
                    value = format_float(s["avg_cont_duration"])

                elif indicator == "Avg. re-entry interval (cycles)":
                    value = format_float(s["avg_gap"])

                elif indicator.startswith("PIs with ≥ "):
                    k = int(indicator.split("≥ ")[1].split(" ")[0])
                    num, pct = s["consecutive_stats"][k]
                    value = format_count_pct(num, pct)

                elif indicator == "Max Continuous Duration":
                    value = format_max_spell(s["max_spell"], s["num_holders"])

                else:
                    value = ""

                col = {
                    1: "1-Year Cycle",
                    3: "3-Year Cycle",
                    5: "5-Year Cycle"
                }[cycle]

                row[col] = value

            rows.append(row)
            first_row = False

    return pd.DataFrame(rows)


# ============================
# 5. Load China and U.S. data
# ============================
df_china = pd.read_excel("China_NSFC_data.xlsx")[["PI", "Organization", "Year", "Funding"]]
df_china["PI_ID"] = df_china["PI"].astype(str) + "_" + df_china["Organization"].astype(str)

df_us = pd.read_excel("US_funding_data.xlsx")[["PI", "Organization", "Year", "Funding"]]
df_us["PI_ID"] = df_us["PI"].astype(str) + "_" + df_us["Organization"].astype(str)

# ============================
# 6. Analyze survival for 1-, 3-, and 5-year cycles
# ============================
china_surv = {}
china_stats = {}
for c in [1, 3, 5]:
    china_surv[c], china_stats[c] = analyze_survival(df_china, c)

us_surv = {}
us_stats = {}
for c in [1, 3, 5]:
    us_surv[c], us_stats[c] = analyze_survival(df_us, c)

# ============================
# 7. Export descriptive statistics table
# ============================
table1 = build_descriptive_table(china_stats, us_stats)

table1_path = os.path.join("tables", "Table1_Descriptive_Statistics_Top10_Funding_Persistence_1990_2020.xlsx")
table1.to_excel(table1_path, index=False)

print(f"Table 1 saved to: {table1_path}")

# ============================
# 8. Plot Kaplan-Meier survival curves
# ============================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
colors = {"yearly": "#7F91B9", "3yr": "#BACBE0", "5yr": "#CDCECD"}

def plot_km(ax, survival_dict, country_label):
    kmf = KaplanMeierFitter()

    for (data, label, mult, color, ypos) in [
        (survival_dict[1], "Yearly Top 10%", 1, colors["yearly"], 0.50),
        (survival_dict[3], "3-Year Period Top 10%", 3, colors["3yr"], 0.45),
        (survival_dict[5], "5-Year Period Top 10%", 5, colors["5yr"], 0.40)
    ]:
        kmf.fit(data["T"] * mult, event_observed=data["E"], label=label)
        x = kmf.survival_function_.index.values
        y = kmf.survival_function_[label].values

        ax.step(x, y, where="post", lw=2, color=color, label=label)
        ax.fill_between(x, y, step="post", alpha=0.3, color=color)
        ax.axvline(kmf.median_survival_time_, color=color, ls="--", lw=1)

        ax.text(
            kmf.median_survival_time_,
            ypos,
            f"Median = {kmf.median_survival_time_:.1f} years",
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
            color=color
        )

    ax.set_title(country_label, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time since first entry into Top 10% (years)", fontsize=12)
    ax.grid(True, ls=":", color="gray", alpha=0.5)

# Left panel: United States
plot_km(axes[0], us_surv, "(a) United States")
axes[0].set_ylabel("Survival probability (remaining in Top 10%)", fontsize=12)
axes[0].legend(frameon=False, loc="upper right")

# Right panel: China
plot_km(axes[1], china_surv, "(b) China")
axes[1].legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)

figure_path = os.path.join("figures", "Figure7_Kaplan_Meier_Survival_Top10_US_China.png")
plt.savefig(figure_path, dpi=400, bbox_inches="tight")
# plt.show()

print(f"Figure 7 saved to: {figure_path}")