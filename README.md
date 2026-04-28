# Clinical Feature Selection Pipeline

This repository contains a modular Python pipeline for dimensionality reduction, specifically designed for clinical questionnaires and datasets. The workflow safely reduces the number of features while ensuring clinical relevance and high predictive performance.

## 🗂️ Repository Navigation

The project is organized into modular directories to separate raw data, processing scripts, and execution code.

* **`data/`**: Designated for your datasets. *(Note: This folder is ignored by Git to prevent pushing sensitive clinical data to the web).*
    * `raw/`: Place your initial, unmodified datasets here.
    * `processed/`: The pipeline will export optimized feature subsets here for clinical review.
* **`src/`**: Contains the core, reusable Python modules.
    * `preprocessing.py`: Handles unsupervised data cleaning (variance filtering and correlation analysis).
    * `modeling.py`: Manages the supervised machine learning steps (train-test splitting and SHAP-based recursive feature elimination).
* **`notebooks/`**: A workspace for Jupyter notebooks. Use this for exploratory data analysis (EDA) or presenting the final feature subset to clinicians.
* **`main.py`**: The central execution script that ties the `src/` modules together into a single, automated workflow.
* **`requirements.txt`**: A list of all Python packages required to run the pipeline.

## ⚙️ The 5-Step Workflow

This pipeline automates the mathematical and machine learning aspects of feature selection, pausing where human clinical expertise is required.

1.  **Variance Pre-filtering**: Removes features with near-zero variance (e.g., a questionnaire item where 99% of patients answered "Yes").
2.  **Correlation Analysis**: Computes Spearman correlations to flag highly redundant features for clinician review, preventing multicollinearity.
3.  **Train-Test Split**: Isolates a hold-out test set to ensure final model validation is unbiased. 
4.  **SHAP-RFECV**: Runs inside a Cross-Validation loop on the training data. It uses an XGBoost model and SHAP (SHapley Additive exPlanations) values to iteratively drop the least important features while tracking model performance.
5.  **Final Clinical Review**: Exports the mathematically optimized feature list so clinicians can verify that the reduced questionnaire still makes clinical sense.

## 🚀 Setup and Installation

To use this repository locally, we recommend setting up a virtual environment. 

**1. Clone the repository:**
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/YOUR_USERNAME/clinical-feature-selection.git](https://github.com/YOUR_USERNAME/clinical-feature-selection.git)
cd clinical-feature-selection
