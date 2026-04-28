import pandas as pd
from src.preprocessing import drop_low_variance_features, get_highly_correlated_features
from src.modeling import perform_train_test_split, custom_shap_rfecv

def run_pipeline():
    # 0. Load Data (Replace with your actual data loading)
    # df = pd.read_csv('data/raw/clinical_data.csv')
    
    # --- Dummy Data for demonstration ---
    print("Loading data...")
    import numpy as np
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(200, 20), columns=[f'feature_{i}' for i in range(20)])
    df['target'] = np.random.randint(0, 2, 200)
    df['zero_var_feature'] = 1  # Add a dummy zero-variance feature
    # ------------------------------------

    X = df.drop(columns=['target'])
    y = df['target']

    # Step 1: Variance Pre-filtering
    X_filtered = drop_low_variance_features(X, threshold=0.0)
    
    # Step 2: Correlation Analysis (Save list for Clinician Review)
    redundant_features = get_highly_correlated_features(X_filtered, threshold=0.85)
    
    # Note: Clinicians review `redundant_features` here. 
    # For the automated pipeline, we assume they approved dropping them:
    X_filtered = X_filtered.drop(columns=redundant_features)
    
    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = perform_train_test_split(X_filtered, y)
    
    # Step 4: SHAP-RFECV
    optimal_features, cv_history = custom_shap_rfecv(X_train, y_train, cv_splits=5, step=1)
    
    # Step 5: Final Clinical Review
    print("\n--- Step 5: Final Clinical Review ---")
    print("Exporting optimized feature subset for clinician review...")
    
    final_subset_df = X_train[optimal_features]
    # final_subset_df.to_csv('data/processed/optimal_features_for_review.csv', index=False)
    
    print("Pipeline completed successfully. Please check data/processed/ for outputs.")

if __name__ == "__main__":
    run_pipeline()