import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, age_cap=80.0, income_cap_percentile=99.0):
        self.age_cap = age_cap
        self.income_cap_percentile = income_cap_percentile
        self.income_cap_ = None
        
    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=['Age', 'Income', 'LoanAmount', 'CreditScore'])
            
        self.income_cap_ = X['Income'].quantile(self.income_cap_percentile / 100.0)
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        if isinstance(X_copy, np.ndarray):
            X_copy = pd.DataFrame(X_copy, columns=['Age', 'Income', 'LoanAmount', 'CreditScore'])
            
        X_copy.loc[X_copy['Age'] > 100.0, 'Age'] = self.age_cap
        X_copy['Age'] = np.clip(X_copy['Age'], None, self.age_cap)
        X_copy['Income'] = np.clip(X_copy['Income'], None, self.income_cap_)
        
        return X_copy

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=['Age', 'Income', 'LoanAmount', 'CreditScore'])
            
        X_copy = X.copy()
        X_copy['DebtToIncomeRatio'] = X_copy['LoanAmount'] / X_copy['Income']
        
        return X_copy

def get_preprocessing_pipeline():
    preprocessor = Pipeline(steps=[
        ('outliers', OutlierHandler()),
        ('features', FeatureEngineer()),
        ('scaler', StandardScaler())
    ])
    return preprocessor

if __name__ == "__main__":
    print("Testing preprocessing pipeline...")
    data = pd.DataFrame({
        'Age': [25, 144, 45, 120, 35],
        'Income': [50000, 1500000, 75000, 80000, 1000000],
        'LoanAmount': [10000, 300000, 15000, 20000, 250000],
        'CreditScore': [600, 750, 700, 650, 800]
    })
    
    pipeline = get_preprocessing_pipeline()
    pipeline.fit(data)
    transformed = pipeline.transform(data)
    
    print("\nOriginal Data:")
    print(data)
    print("\nTransformed Data (Scaled & Features Added):")
    print(transformed)
    print("\nFitted Income Cap:", pipeline.named_steps['outliers'].income_cap_)
