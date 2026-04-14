# Public Research Funding in China and the United States

This repository contains the code and data used to reproduce the empirical results in the manuscript:

**Public Research Funding in China and the United States: Inequality, Mobility, and Elite Persistence, 1990–2020**

---

## Overview

This study examines the allocation dynamics of public research funding in China and the United States from 1990 to 2020. Using project-level data from the National Natural Science Foundation of China (NSFC) and the U.S. National Science Foundation (NSF), the analysis focuses on three key dimensions of funding regimes:

* **Inequality** (Gini coefficients and Top-k funding shares)
* **Mobility** (funding-quantile transition matrices and mobility indices)
* **Elite persistence** (Kaplan–Meier survival analysis of top-decile researchers)

The results show that similar levels of aggregate inequality can coexist with different degrees of mobility and elite persistence, highlighting the importance of considering dynamic processes beyond static concentration.

---

## Repository Structure

```
.
├── README.md
├── China_NSFC_data.xlsx
├── US_funding_data.xlsx
├── figure1_funding_descriptive_statistics.py
├── figure2_gini_analysis.py
├── figure3_topk_shares.py
├── figure4_5_6_transition_matrix_and_mobility.py
├── figure7_table1_survival_analysis.py
├── figures/
├── tables/
```

* `*.py` : Python scripts for generating figures and tables
* `figures/` : exported figures (PNG format)
* `tables/` : exported tables (Excel format)

---

## Data Sources

The empirical analysis is based on publicly available research funding data:

* **United States (NSF)**
  NSF Award Search Database
  https://www.nsf.gov/awardsearch/

* **China (NSFC)**
  Izaiwen Research Data Platform
  https://www.izaiwen.cn/pro

The datasets include:

* Principal Investigator (PI)
* Institutional affiliation
* Year of award
* Funding amount

---

## Requirements

This project uses **Python 3.9**.

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn lifelines openpyxl
```

---

## Reproduction Guide

Run the scripts in the following order to reproduce all figures and tables:

1. `figure1_funding_descriptive_statistics.py`
   → Funding trends and project counts

2. `figure2_gini_analysis.py`
   → Multi-level Gini coefficients

3. `figure3_topk_shares.py`
   → Top-k funding share distributions

4. `figure4_5_6_transition_matrix_and_mobility.py`
   → Transition matrices and mobility indices

5. `figure7_table1_survival_analysis.py`
   → Kaplan–Meier survival analysis and Table 1

---

## Output

* Figures are saved in:
  `figures/`

* Tables are saved in:
  `tables/`

Including:

* **Figure 1–7**
* **Table 1: Descriptive statistics of Top 10% funding persistence (1990–2020)**

---

## Notes

* All analyses are conducted using each country's original currency.
* The comparison focuses on **relative funding distributions**, not absolute monetary values.
* Some datasets may not be redistributed due to licensing or access restrictions. Users should obtain the original data from the sources listed above.

---

## Citation

If you use this repository, please cite:

> Public Research Funding in China and the United States: Inequality, Mobility, and Elite Persistence, 1990–2020

---

## Contact

For questions or replication issues, please contact the author.
