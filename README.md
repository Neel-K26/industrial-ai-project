# Agentic Multi-Machine Predictive Maintenance (AMMPM)
### Using Time-Series Foundation Models for Explainable and Trustworthy RUL Prediction

> **Institution:** Pimpri Chinchwad University (PCU), School of Engineering Technology, CSE (AI & ML)  
> **Research Supervisor:** Dr. Rahul Sonkamble  
> **Team:** Neel Khairnar (Lead), Ronak Patil, Yash Bias, Shivam Parashar  
> **Target Venues:** IEEE Transactions on Industrial Informatics · IEEE PHM 2026 · IEEE CASE 2026

---

## Research Objective

This project develops a **trustworthy predictive maintenance framework** that goes beyond single-asset, accuracy-only approaches. The core thesis: industrial AI systems must simultaneously provide accurate RUL predictions, explain *why* a failure is predicted, and express *how confident* they are — across heterogeneous machine types, without retraining from scratch.

**Three research contributions being validated:**
1. Cross-machine transfer using Time-Series Foundation Models (TSFMs)
2. Quantitative XAI faithfulness via SHAP attribution stability
3. Uncertainty calibration under domain shift

---

## Current Status: Benchmark Phase Complete (E1–E4)

### Results Table — C-MAPSS FD001 (100 test engines, seed 42)

| Exp | Model | Supervision | RMSE ↓ | MAE ↓ | PHM Score ↓ |
|-----|-------|-------------|--------|-------|-------------|
| E1 | LSTM (2-layer, hidden=64) | Full | 40.53 | 35.10 | 18182.32 |
| E2 | Transformer Encoder (d=64, heads=4) | Full | **14.62** | **10.85** | 412.32 |
| E3 | PatchTST (patch=6, stride=3) | Full | 14.83 | 11.97 | **385.59** |
| E4 | Chronos (zero-shot, unsupervised) | None | 45.37 | 39.16 | 23919.40 |

> **PHM Score** = asymmetric penalty from the PHM08 challenge. Late predictions are penalized more heavily than early ones (exp(d/10)-1 vs exp(-d/13)-1). Lower is safer.

### Key Findings So Far

- **E1→E2:** Attention mechanism drops RMSE 64% over recurrent baseline on identical pipeline — attributable purely to encoder architecture.
- **E2≈E3:** Patch tokenization offers no advantage on short (30-cycle) single-condition windows. PatchTST's strength emerges on longer, multi-condition sequences — to be tested on FD002/FD004 and XJTU-SY.
- **E4:** A fully unsupervised TSFM (Chronos uncertainty heuristic, zero labels) scores comparably to the LSTM baseline on RMSE but significantly worse on PHM Score, confirming the research gap: TSFMs need task-aware adaptation to be industrially deployable. This motivates E5 (fine-tuned Chronos).

---

## Repository Structure

```
industrial-ai-project/
│
├── notebooks/                        # Experiment notebooks (one per experiment)
│   ├── E1_lstm_baseline.ipynb        # ✅ LSTM baseline — RMSE 40.53
│   ├── E2_transformer_baseline.ipynb # ✅ Transformer — RMSE 14.62
│   ├── E3_patchtst_baseline.ipynb    # ✅ PatchTST — RMSE 14.83
│   ├── E4_chronos_zeroshot.ipynb     # ✅ Chronos zero-shot — RMSE 45.37
│   └── EX_visualizations.ipynb      # ✅ Publication figures (6 figures)
│
├── src/
│   ├── data/
│   │   └── cmapss_loader.py          # NASA C-MAPSS download + preprocessing pipeline
│   ├── utils/
│   │   └── seed.py                   # Global seed (42) for full reproducibility
│   ├── models/                       # Model definitions (populated by E5+)
│   ├── agents/                       # LangGraph agents (populated by E8)
│   └── evaluation/                   # Metrics and evaluation utilities
│
├── configs/
│   ├── experiment_config.yaml        # 8 experiments, 6 metrics, 6 models defined
│   └── paths.py                      # All paths resolved from project root
│
├── results/
│   ├── E1_lstm_fd001.json            # ✅ Metrics + reproducibility confirmed
│   ├── E2_transformer_fd001.json     # ✅
│   ├── E3_patchtst_fd001.json        # ✅
│   ├── E4_chronos_zeroshot_fd001.json# ✅
│   ├── *_predictions.npz             # Per-engine predictions (all 100 engines)
│   └── figures/
│       ├── fig1_lollipop.png         # RMSE progression across model families
│       ├── fig2_phm_bar.png          # PHM risk severity (log scale)
│       ├── fig3_scatter_grid.png     # Predicted vs true RUL (2×2 panel)
│       ├── fig4_error_violin.png     # Error distribution + bias direction
│       ├── fig5_supervision_bubble.png # Supervision vs performance (research gap)
│       └── fig6_radar.png            # Composite accuracy/efficiency radar
│
├── data/
│   └── raw/cmapss/                   # NASA C-MAPSS FD001–FD004 (gitignored)
│
├── requirements.txt                  # All dependencies pinned
└── .gitignore                        # venv/, data/raw/, *.so excluded
```

---

## Datasets

| Dataset | Purpose | Status |
|---------|---------|--------|
| NASA C-MAPSS FD001–FD004 | Primary RUL benchmarking (turbofan engines) | ✅ Downloaded & verified |
| XJTU-SY Bearing | Cross-machine transfer (bearing domain) | ⏳ E5+ |
| AI4I 2020 | Failure-cause explainability alignment | ⏳ E6+ |

**C-MAPSS preprocessing:** min-max normalization (train-fit only), RUL capped at 125 cycles, sliding window length 30, engine-level 80/20 train/val split, seed 42 throughout.

---

## Reproducibility

Every experiment is fully reproducible:
- Global seed 42 set via `src/utils/seed.py` (covers `random`, `numpy`, `torch`, CUDA)
- All hyperparameters defined in `configs/experiment_config.yaml`
- Results verified by running each notebook twice — byte-identical metrics confirmed
- No data leakage: normalization fitted on training engines only

---

## Experiment Roadmap

| Exp | Description | Status |
|-----|-------------|--------|
| E1 | LSTM baseline (FD001) | ✅ Complete |
| E2 | Transformer baseline (FD001) | ✅ Complete |
| E3 | PatchTST / TSFM-style (FD001) | ✅ Complete |
| E4 | Chronos zero-shot (FD001) | ✅ Complete |
| E5 | Chronos fine-tuned (FD001 → XJTU-SY transfer) | 🔄 Next |
| E6 | SHAP explainability + attribution stability | ⏳ Pending |
| E7 | Uncertainty quantification + calibration | ⏳ Pending |
| E8 | Full LangGraph agentic system | ⏳ Pending |

---

## Technology Stack

| Category | Tools |
|----------|-------|
| Core ML | PyTorch, Scikit-learn |
| TSFMs | Chronos (amazon/chronos-t5-small), PatchTST |
| Explainability | SHAP |
| Agentic Orchestration | LangGraph 0.2.55 |
| Data | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| Experiment Tracking | JSON results + NPZ predictions per run |
| Dev Environment | Python 3.11, VS Code, Claude Code |
| Version Control | Git + GitHub |

---

## Setup

```bash
# Clone and setup
git clone https://github.com/Neel-K26/industrial-ai-project.git
cd industrial-ai-project
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download C-MAPSS data
python src/data/cmapss_loader.py

# Run experiments in order
jupyter notebook notebooks/E1_lstm_baseline.ipynb
```

---

## Non-Negotiable Scope Boundaries

This project explicitly **excludes** to maintain focus:
- Physics-informed Neural Networks (future scope)
- Continual / online learning
- Knowledge graphs
- Financial or medical time-series domains
- LLM agents performing numerical reasoning (LangGraph is deterministic Python orchestration only)

---

## Publication Target

**Primary:** IEEE Transactions on Industrial Informatics (Q1)  
**Conference:** IEEE PHM 2026 / IEEE CASE 2026  
**Arxiv preprint:** Planned post E7 completion

**Core claim for submission:** First unified framework jointly evaluating cross-machine TSFM transfer, XAI faithfulness, and uncertainty calibration under domain shift — on three heterogeneous industrial benchmark datasets.
