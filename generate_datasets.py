import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

np.random.seed(42)

def make_classification_dataset():
    # Base dataset
    X, y = make_classification(n_samples=800, n_features=10, n_informative=6, n_redundant=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(1, 11)])
    df['Target_Class'] = y
    
    df['Categorical_Col'] = np.random.choice(['Red', 'Green', 'Blue'], size=len(df))
    
    # Missing values to showcase imputation
    for col in ['Feature_3', 'Feature_7']:
        df.loc[np.random.rand(len(df)) < 0.05, col] = np.nan
        
    # Outliers to showcase outlier detection
    outliers = np.random.choice(df.index, size=20, replace=False)
    df.loc[outliers, 'Feature_1'] = df.loc[outliers, 'Feature_1'] * 15
    
    df.to_csv('showcase_classification.csv', index=False)
    print("Created: showcase_classification.csv")

def make_regression_dataset():
    # Base dataset
    X, y = make_regression(n_samples=800, n_features=10, noise=2.0, random_state=42)
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(1, 11)])
    df['Target_Value'] = y
    
    df['Categorical_Col'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df))
    
    # Missing values to showcase imputation
    for col in ['Feature_2', 'Feature_6']:
        df.loc[np.random.rand(len(df)) < 0.05, col] = np.nan
        
    # Outliers to showcase outlier detection
    outliers = np.random.choice(df.index, size=20, replace=False)
    df.loc[outliers, 'Feature_5'] = df.loc[outliers, 'Feature_5'] * 20
    
    df.to_csv('showcase_regression.csv', index=False)
    print("Created: showcase_regression.csv")

make_classification_dataset()
make_regression_dataset()
