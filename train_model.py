import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib
from preprocessing import get_preprocessing_pipeline

def train_loan_model(data_path="loan_data.csv", model_path="loan_model.pkl"):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    X = df[['Age', 'Income', 'LoanAmount', 'CreditScore']]
    y = df['Approved']
    
    print("Splitting data into train and test sets (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training shape: {X_train.shape}, Class distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Testing shape: {X_test.shape}, Class distribution:\n{y_test.value_counts(normalize=True)}")
    
    preprocessor = get_preprocessing_pipeline()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__max_depth': [3, 4, 5, 7, 10, None],
        'classifier__min_samples_split': [2, 5, 10, 20],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__class_weight': ['balanced', None]
    }
    
    print("Initializing GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("Fitting Grid Search on training data...")
    grid_search.fit(X_train, y_train)
    
    print("\n=== Grid Search Results ===")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
    
    best_pipeline = grid_search.best_estimator_
    
    train_acc = best_pipeline.score(X_train, y_train)
    print(f"Best Model Accuracy on Training Set: {train_acc:.4f}")
    
    print(f"Saving final model to '{model_path}'...")
    joblib.dump(best_pipeline, model_path)
    print("Model saved successfully!")
    
    loaded_pipeline = joblib.load(model_path)
    sample_pred = loaded_pipeline.predict(X_test.head(2))
    print(f"Verification prediction on first 2 test rows: {sample_pred}")

if __name__ == "__main__":
    train_loan_model()
