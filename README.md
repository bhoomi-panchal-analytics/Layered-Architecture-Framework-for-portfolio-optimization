# Layered-Architecture-Framework-for-portfolio-optimization
Integrated Multi-Framework Portfolio Optimization Engine with Fundamental, ESG, and Strategic Game-Theoretic Layers
A multi-layer quantitative investment research framework integrating advanced risk modeling (GARCH family), fundamental valuation models, ESG carbon exposure metrics, and robust portfolio optimization techniques including Black–Litterman, entropy pooling, and game-theoretic allocation. Built with a full-stack Streamlit interface and institutional-grade backtesting architecture.

Overview

This project presents a comprehensive, research-oriented portfolio construction framework that integrates advanced asset pricing, corporate valuation, ESG transition risk, and dynamic risk modeling into a unified optimization engine.

The system moves beyond classical mean-variance optimization by combining:

• Bayesian return adjustment via Black–Litterman
• Distributional robustness using entropy pooling
• Strategic allocation through game-theoretic modeling
• Carbon transition risk integration (CBAM exposure & carbon credit sensitivity)
• Advanced volatility forecasting using GARCH-family models

The objective is to construct portfolios that are statistically robust, economically grounded, and ESG-aware under multiple macroeconomic regimes.



# Regime-Aware Tail-Risk Constrained Portfolio System

## Overview

This project implements a hierarchical, governance-first portfolio allocation framework designed to enhance capital preservation under structural market uncertainty.

The system integrates:

• Macro regime detection (Hidden Markov Model)
• Regime-dependent volatility modeling (GARCH / MS-GARCH approximation)
• Systemic contagion monitoring (rolling correlation networks)
• Tail-risk diagnostics (VaR, CVaR, fat-tail analysis)
• Multi-strategy portfolio optimization (Minimum Variance, Max Sharpe, Risk Parity)
• Interactive institutional-style dashboard (Streamlit)

The objective is not short-term return prediction.
The objective is structural survivability across market regimes.

---

## Core Architecture

The system is structured hierarchically:

Layer 1 – Macro Regime Detection
Identifies probabilistic market states (expansion, transition, crisis).

Layer 2 – Volatility Modeling
Estimates regime-dependent conditional variance.

Layer 3 – Contagion Monitoring
Detects correlation convergence and systemic stress.

Layer 4 – Tail-Risk Governance
Applies VaR and CVaR constraints.

Layer 5 – Allocation Engine
Allocates capital within governance boundaries.

Higher layers override lower layers.
Risk governance dominates optimization.

---

## Data Inputs

The system uses multi-asset and macro datasets, including:

• SPY, TLT, GLD, DBC, UUP, SHY
• VIX index
• Yield curve spread
• Credit spreads
• Financial stress index
• Synthetic stress datasets (optional)

All CSV files are placed inside:

```
/data/
```

---

## Folder Structure

```
project/
│
├── data/
│   ├── market_data.csv
│   ├── regime_probabilities.csv
│   ├── garch.csv
│   ├── contagion.csv
│   └── additional macro datasets
│
├── utils/
│   ├── load_data.py
│   ├── feature_engineering.py
│
├── pages/
│   ├── 1_Macro_Regime.py
│   ├── 2_Volatility.py
│   ├── 3_Contagion.py
│   ├── 4_Portfolio_Allocation.py
│   ├── 5_Conclusion.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Libraries Used

Core Libraries:

• pandas
• numpy
• plotly
• streamlit
• statsmodels
• arch
• hmmlearn
• networkx
• scipy

Optional (advanced modeling):

• scikit-learn
• matplotlib

All dependencies are listed in:

```
requirements.txt
```

Install using:

```bash
pip install -r requirements.txt
```

---

## How to Run the Model

### 1. Clone the Repository

```bash
git clone <repo_url>
cd project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Data

Place required CSV files inside the `/data/` folder.

Minimum required:

• market_data.csv
• regime_probabilities.csv
• garch.csv
• contagion.csv

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

The system will open in your browser.

---

## Dashboard Pages

Macro Regime Page
• Regime probability heatmap
• Crisis timeline overlay
• Regime confidence metrics

Volatility Page
• MS-GARCH volatility forecast
• VIX comparison
• Volatility clustering visualization

Contagion Page
• Rolling correlation matrix
• Network density timeline
• Systemic stress index

Portfolio Allocation Page
• Asset selection
• Optimization mode selection
• VaR / CVaR diagnostics
• Diversification ratio
• Fat-tail analysis
• Drawdown monitoring

Conclusion Page
• System summary
• Architecture explanation
• Survivability analysis

---

## What This System Does

• Detects structural market regimes
• Models volatility clustering
• Quantifies systemic stress
• Constrains tail risk
• Allocates capital under governance hierarchy
• Provides institutional-style monitoring dashboard

---

## What This System Does NOT Do

• Guarantee alpha
• Eliminate drawdowns
• Time market tops and bottoms
• Provide financial advice
• Replace live risk management systems

It is a research-grade decision-support framework.

---

## Research Applications

This framework can be extended for:

• Regime-conditioned factor tilting
• Robust Bayesian allocation
• Extreme Value Theory tail modeling
• Reinforcement learning under CVaR constraints
• Institutional stress testing

---

## Deployment

The application can be deployed using:

• Streamlit Cloud
• Docker container
• AWS EC2
• Local development server

---

## Design Philosophy

The system prioritizes:

Survivability > Optimization Sensitivity
Governance > Prediction
Tail Control > Variance Minimization

Long-term compounding depends on drawdown control, not peak return.

---

## Disclaimer

This project is for research and educational purposes only.
It does not constitute financial advice.


