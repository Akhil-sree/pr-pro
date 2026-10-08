# CATSSR-BN: Aviation Safety Risk Analysis with Bayesian Networks

Reproduction and extension of **Zhou et al. (2022), "Civil aviation safety assessment
based on a Bayesian network", _Safety Science_ 157, 105942** — learning a Bayesian
network over aviation incident causal factors and result events, then using it for
risk ranking and safety-budget allocation.

## Overview

- **Problem:** which causal factors (equipment / human / weather events) drive the
  riskiest incident outcomes, and where should a limited safety budget go?
- **Data:** NASA ASRS incident reports, Jan 2017 – Dec 2019
  (`data/raw/ASRS_2017_2019.xlsx`, ~12 MB, committed to this repo).
- **Methods:** binary incident × event matrix → node-family classification
  (causal AE/HE/WE vs. result RE) → MMHC structure learning with a
  no-result→result blacklist → Bayesian parameter fitting (CPTs) → 10-fold
  cross-validation (log-likelihood loss, MMHC vs. RSMAX2) → mutual-information
  ranking → risk coefficients and greedy budget optimisation → network plots.
- **Outputs:** learned edges, CPTs, CV losses, MI rankings, risk tables, network
  figures (written next to the notebook on each run; key result tables/figures
  are described below).

## Project structure

```text
.
├── README.md
├── requirements.txt          # Python deps (only what the notebook imports)
├── .gitignore
├── LICENSE                   # MIT
├── notebooks/
│   └── catssr_bn_analysis.ipynb   # main notebook (run from this folder)
├── R/
│   ├── catssr_bn.R           # BN pipeline: MMHC → CPTs → CV → MI → plot
│   └── requirements.R        # install.packages(c("bnlearn", "igraph"))
├── data/
│   ├── raw/ASRS_2017_2019.xlsx    # source dataset (committed, ~12 MB)
│   └── processed/            # for derived matrices (empty until you run)
└── outputs/
    ├── figures/ tables/ reports/  # suggested homes for saved results
```

## Methodology (beginner-friendly)

- **Causal vs. result node families:** every event code is either a *cause*
  (AE = aircraft/environment anomaly, HE = human factor, WE = weather/wake) or a
  *result* (RE = what happened, e.g. go-around, injury). Results can only be
  *caused*, never cause each other — enforced as a blacklist during learning.
- **Bayesian network:** a directed graph where an edge A → B means "A influences
  B", plus a probability table (CPT) per node quantifying that influence.
- **MMHC structure learning:** a hybrid algorithm (constraint + score search)
  that discovers the graph from data; this is the algorithm the paper used.
- **Bayesian parameter learning:** fills in every CPT with smoothed probability
  estimates (`bn.fit(..., method = "bayes")`).
- **Cross-validation:** 10-fold log-likelihood loss checks the model generalises;
  MMHC is compared against RSMAX2 exactly as in the paper's Fig. 3.
- **Mutual information (MI):** measures how much knowing a causal factor tells
  you about a result event — used to rank "top causal factors" per high-risk
  outcome (paper Tables 3–4).
- **Risk analysis:** risk coefficients from conditional-probability gaps feed a
  greedy budget allocator (marginal risk reduction per unit cost).

## Installation

### Python

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

Needs Python 3 with `pandas numpy matplotlib networkx openpyxl ipython`.
(`openpyxl` is required to read the `.xlsx` dataset.)

### R

```r
source("R/requirements.R")
```

Only two packages are used: `bnlearn` (structure/parameter learning) and
`igraph` (network plot). Any recent R (≥ 4.0) works.

## How to run

The notebook must run **before / around** the R script: notebook cell 2 builds
`bn_matrix.csv` + `node_family.csv`, which `R/catssr_bn.R` reads. Run everything
with `notebooks/` as the working directory:

1. Clone the repository.
2. Install Python deps (`pip install -r requirements.txt`) and R deps
   (`source("R/requirements.R")`).
3. `cd notebooks/`
4. Open `catssr_bn_analysis.ipynb` and run cells **in order**:
   - Cells 0–1 are **Google Colab only** (installs `r-base`, uploads files).
     Locally: skip them — the dataset and R script are already in the repo.
   - Cell 2 builds the binary matrix (`bn_matrix.csv`, `node_family.csv`,
     8913 incidents × 96 nodes).
   - Cell 3 runs `Rscript ../R/catssr_bn.R` (MMHC → CPTs → CV → MI → plot).
   - Cells 4–7 compute risk priority and (interactive) budget allocation.
   - Cells 9–17 (paper-checks section) rebuild the matrix on the paper's
     Appendix I/II taxonomy (`bn_matrix_paper.csv`, 8911 rows) and reproduce
     Tables 1–5 / Figs. 1, 3–6 plus sensitivity checks. These need R with
     `bnlearn` available as `Rscript`.
5. Inspect generated CSVs/PNGs in `notebooks/` (move the ones you want to keep
   into `outputs/tables/` or `outputs/figures/`).

`R/catssr_bn.R` can also be run standalone once `bn_matrix.csv` and
`node_family.csv` exist beside it.

## Outputs

Generated at runtime (not committed); each is written beside the notebook:

| File | Meaning |
|---|---|
| `bn_matrix.csv` / `node_family.csv` | binary incident × event matrix + causal/result labels (input to R) |
| `bn_edges.csv` | learned DAG edge list (parent → child) |
| `cpts_export.csv` | every node's conditional probability table, long format |
| `kfold_logloss.csv` | per-fold log-likelihood loss |
| `mmhc_vs_rsmax2.csv` | mean/sd log-loss comparison (paper Fig. 3) |
| `mutual_information_ranking.csv` | MI per causal → result pair |
| `prior_probabilities.csv` | prior P(E=1) per causal factor (paper Table 1 style) |
| `table3_high_risk_causal_factors.csv` | top-5 causal factors per high-risk result (RE04/RE05/RE14/RE29) |
| `risk_priority_order.csv` | risk coefficients + baseline scores for budget allocation |
| `paper_*.csv` / `paper_fig*.png` | paper-taxonomy rerun: Tables 1–5, Figs. 1/3–6, CV details, sensitivity |
| `catssr_bn_network.png` | network diagram (node size = degree, colour = family) |

## Reproducibility notes

- Random seeds: inline paper-check R code uses `set.seed(42)`; the network
  plot in `R/catssr_bn.R` uses `set.seed(42)` for layout. `R/catssr_bn.R` sets
  no learning seed (MMHC/BDeu are deterministic given data).
- The paper's exact 7,265-incident subset and some constants (W\*, Table 3
  denominators, Table 4 definition) are not recoverable from the paper text —
  the notebook documents these gaps honestly (cells 8–9 markdown, cell 16
  definition checks) instead of guessing.
- Fig. 4 risk levels for non-high events are author-assumed, flagged as such in
  the notebook — replace with the published table if available.
- Reference: Zhou et al. (2022), *Safety Science* 157, 105942.
