# Vironix MPI 2025 group

## What is this?
code/analysis relating to questions of synthetic generation and clinician evaluation of real/synthetic patient profiles in the context of Chronic Kidney Disease (CKD) longitudinal profiles.

### Problem Statement (2025)

1. **Enhanced Synthetic Data Generalization**
   - Embed mechanisms in generative models to simulate acute kidney injury (AKI) episodes (i.e., rapid CKD stage spikes and subsequent reversion) so synthetic trajectories reflect both chronic progression and clinically treated, reversible events. 

2. **Composite Metric**  
   - Develop and validate a **Clinical Fidelity Index** combining clinician rating and statistical realism.  

3. **Normalization & Generalization**  
   - Adjust for rating variability due to differences in clinician practices (e.g., regional population, clinician experience, practice-based biases)
   - Design metrics that generalize to other chronic conditions (e.g., CHF, COPD).


## Coding Guidelines

* **Languages & Libraries**:

  * Python ≥3.8
  * Core: `numpy`, `pandas`, `scikit-learn`, `PyTorch`
  * Statistical/Bayesian: `statsmodels`, `PyMC3`/`Pyro`
  * Visualization: `matplotlib`, `seaborn`

* **Style**:

  * PEP8 compliance, docstrings for all functions (`numpy` style).

## Getting started
???
