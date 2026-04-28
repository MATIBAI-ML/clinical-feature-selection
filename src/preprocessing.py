import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def drop_low_variance_features(df, threshold=0.01):
    """
    Step 1: Remove features with near-zero variance.
    """
    print("Running Variance Pre-filtering...")
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[numeric_cols])
    
    # Get features to keep
    features_to_keep = numeric_cols[selector.get_support(indices=True)]
    dropped_features = set(numeric_cols) - set(features_to_keep)
    
    print(f"Dropped {len(dropped_features)} low-variance features.")
    
    # Return dataframe with kept numeric columns + any categorical columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    final_cols = list(features_to_keep) + list(categorical_cols)
    
    return df[final_cols]

def get_highly_correlated_features(df, threshold=0.85):
    """
    Step 2: Compute Spearman correlation and flag redundant features.
    Returns a list of features recommended for clinician review/removal.
    """
    print("Running Spearman Correlation Analysis...")
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate Spearman correlation matrix
    corr_matrix = numeric_df.corr(method='spearman').abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Found {len(to_drop)} highly correlated features for clinician review.")
    return to_drop