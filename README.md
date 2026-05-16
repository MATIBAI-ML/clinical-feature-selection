# Clinical Feature Selection Pipeline

This repository contains our proposed workflow for feature selection, specifically designed for clinical questionnaires and datasets. 

## 🗂️ Repository Navigation

The project is organized into directories to separate raw data, processing scripts, and execution code.

* **`data/`**: Designated for our datasets. *(Note: This folder is ignored by Git to prevent pushing sensitive clinical data to the web).*
    * `raw/`: Place the initial, unmodified datasets here.
    * `processed/`: The pipeline will export optimized feature subsets here for clinical review.
* **`src/`**: Contains the core, reusable Python modules.
    * `preprocessing.py`: Handles unsupervised data cleaning (variance filtering and correlation analysis).
    * `modeling.py`: Manages the supervised machine learning steps (train-test splitting and SHAP-based recursive feature elimination).
* **`notebooks/`**: A workspace for Jupyter notebooks. Use this for exploratory data analysis (EDA) or presenting the final feature subset to clinicians.
* **`main.py`**: The central execution script that ties the `src/` modules together into a single, automated workflow.
* **`requirements.txt`**: A list of all Python packages required to run the pipeline.

## ⚙️ Our Proposed Workflow

1.  **Variance Pre-filtering**: Removes features with near-zero variance (e.g., a questionnaire item where 99% of patients answered "Yes").
2.  **Correlation Analysis**: Computes Spearman correlations to flag highly redundant features for clinician review, preventing multicollinearity.
3.  **Train-Test Split**: Isolates a hold-out test set to ensure final model validation is unbiased. 
4.  **SHAP-RFECV**: Runs inside a Cross-Validation loop on the training data. It uses an XGBoost model and SHAP (SHapley Additive exPlanations) values to iteratively drop the least important features while tracking model performance.
5.  **Final Clinical Review**: Exports the mathematically optimized feature list so clinicians can verify that the reduced questionnaire still makes clinical sense.
