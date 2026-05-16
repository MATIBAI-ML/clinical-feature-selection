import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score # or another metric suitable for your clinical task

def perform_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Step 3: Split the data. The test set will not be touched until the end.
    """
    print("Performing Train-Test Split...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def custom_shap_rfecv(X_train, y_train, cv_splits=5, step=1):
    """
    Step 4: SHAP-RFECV (inside Cross-Validation).
    Iteratively drops the least important features based on mean absolute SHAP values.
    """
    print("Starting SHAP-RFECV...")
    current_features = list(X_train.columns)
    history = []
    
    while len(current_features) > 0:
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        cv_scores = []
        shap_values_list = []
        
        # Cross-validation loop
        for train_idx, val_idx in kf.split(X_train[current_features]):
            X_tr, X_val = X_train[current_features].iloc[train_idx], X_train[current_features].iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model
            #model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
            model.fit(X_tr, y_tr)
            
            # Score
            preds = model.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, preds))
            
            # Calculate SHAP values on validation fold
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            shap_values_list.append(np.abs(shap_values).mean(axis=0))
            
        # Average CV score
        mean_score = np.mean(cv_scores)
        
        # Calculate mean SHAP importance across all folds
        mean_shap_importances = np.mean(shap_values_list, axis=0)
        feature_importance_dict = dict(zip(current_features, mean_shap_importances))
        
        # Record history
        history.append({
            'num_features': len(current_features),
            'cv_score': mean_score,
            'features': current_features.copy()
        })
        
        print(f"Features: {len(current_features)} | CV AUC: {mean_score:.4f}")
        
        if len(current_features) <= step:
            break
            
        # Drop the least important feature(s)
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1])
        features_to_drop = [f[0] for f in sorted_features[:step]]
        current_features = [f for f in current_features if f not in features_to_drop]

    # Find the optimal number of features
    best_iteration = max(history, key=lambda x: x['cv_score'])
    print(f"\nOptimal number of features: {best_iteration['num_features']} (CV AUC: {best_iteration['cv_score']:.4f})")
    
    return best_iteration['features'], history