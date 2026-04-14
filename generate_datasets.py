import pandas as pd
import numpy as np

np.random.seed(42)

def make_classification_dataset():
    diseases = ['Flu', 'Allergy', 'Migraine', 'Food_Poisoning', 'Bronchitis']
    symptom_cols = [
        'Fever', 'Cough', 'Fatigue', 'Headache', 'Nausea',
        'Stomach_Pain', 'Shortness_of_Breath', 'Rash', 'Joint_Pain', 'Sore_Throat'
    ]

    data = []
    for _ in range(800):
        disease = np.random.choice(diseases)
        age_group = np.random.choice(['Child', 'Adult', 'Senior'])
        symptom_pattern = {
            'Flu': [1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
            'Allergy': [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
            'Migraine': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            'Food_Poisoning': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            'Bronchitis': [0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
        }[disease]

        symptoms = [
            max(0, min(1, int(value or 0) if np.random.rand() > 0.1 else np.random.choice([0, 1])))
            for value in symptom_pattern
        ]
        body_temp = 36.5 + np.random.randn() * 0.8 + (1.5 if disease == 'Flu' else 0)
        pain_level = int(min(10, max(0, np.random.normal(4 + sum(symptoms) * 0.8, 1.5))))
        data.append(symptoms + [body_temp, pain_level, age_group, disease])

    columns = symptom_cols + ['Body_Temperature', 'Pain_Level', 'Age_Group', 'Target_Disease']
    df = pd.DataFrame(data, columns=columns)

    for col in ['Headache', 'Nausea', 'Stomach_Pain']:
        df.loc[np.random.rand(len(df)) < 0.05, col] = np.nan

    outliers = np.random.choice(df.index, size=20, replace=False)
    df.loc[outliers, 'Body_Temperature'] = df.loc[outliers, 'Body_Temperature'] + np.random.uniform(3.0, 5.0, size=len(outliers))

    df.to_csv('showcase_classification.csv', index=False)
    print('Created: showcase_classification.csv')

def make_regression_dataset():
    symptom_cols = [
        'Fever_Severity', 'Cough_Severity', 'Fatigue_Severity', 'Headache_Severity',
        'Nausea_Severity', 'Stomach_Pain_Severity', 'Breathlessness_Severity',
        'Rash_Severity', 'Joint_Pain_Severity', 'Sore_Throat_Severity'
    ]

    data = []
    for _ in range(800):
        age_group = np.random.choice(['Child', 'Adult', 'Senior'])
        symptoms = [max(0.0, min(10.0, np.random.normal(3 + np.random.rand() * 2, 2))) for _ in symptom_cols]
        risk_score = float(np.dot(symptoms, [0.12, 0.1, 0.13, 0.11, 0.09, 0.1, 0.14, 0.05, 0.08, 0.08]) + np.random.normal(0, 1.8))
        risk_score = max(0.0, min(100.0, risk_score))
        data.append(symptoms + [age_group, risk_score])

    columns = symptom_cols + ['Age_Group', 'Disease_Risk_Score']
    df = pd.DataFrame(data, columns=columns)

    for col in ['Headache_Severity', 'Nausea_Severity', 'Stomach_Pain_Severity']:
        df.loc[np.random.rand(len(df)) < 0.05, col] = np.nan

    outliers = np.random.choice(df.index, size=20, replace=False)
    df.loc[outliers, 'Cough_Severity'] = df.loc[outliers, 'Cough_Severity'] + np.random.uniform(8.0, 12.0, size=len(outliers))

    df.to_csv('showcase_regression.csv', index=False)
    print('Created: showcase_regression.csv')

make_classification_dataset()
make_regression_dataset()
