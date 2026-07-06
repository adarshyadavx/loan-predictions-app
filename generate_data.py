import numpy as np
import pandas as pd

def generate_synthetic_data(filename="loan_data.csv", num_samples=1000, seed=42):
    np.random.seed(seed)
    
    age = np.random.randint(18, 76, size=num_samples).astype(float)
    income = np.random.lognormal(mean=10.8, sigma=0.5, size=num_samples).astype(float)
    credit_score = np.random.randint(300, 851, size=num_samples).astype(float)
    
    loan_amount = (income * np.random.uniform(0.5, 3.0, size=num_samples)).astype(float)
    loan_amount = np.clip(loan_amount, 5000, 500000)
    
    age_outlier_indices = np.random.choice(num_samples, size=3, replace=False)
    age[age_outlier_indices] = 144.0
    
    income_outlier_indices = np.random.choice(
        [i for i in range(num_samples) if i not in age_outlier_indices], 
        size=5, 
        replace=False
    )
    income[income_outlier_indices] = np.array([1200000.0, 1500000.0, 2300000.0, 950000.0, 1800000.0])
    
    debt_to_income = loan_amount / income
    
    norm_credit = (credit_score - 600) / 150
    norm_income = (income - 60000) / 30000
    norm_loan = (loan_amount - 150000) / 100000
    
    logit = 2.0 * norm_credit + 1.0 * norm_income - 1.5 * norm_loan - 2.5 * debt_to_income
    logit += np.random.normal(0, 1.0, size=num_samples)
    
    prob = 1 / (1 + np.exp(-logit))
    
    target_positive_rate = 0.22
    threshold = np.percentile(prob, 100 * (1 - target_positive_rate))
    
    approved = (prob >= threshold).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'Approved': approved
    })
    
    df['Age'] = df['Age'].astype(int)
    df['Income'] = df['Income'].round(2)
    df['LoanAmount'] = df['LoanAmount'].round(2)
    df['CreditScore'] = df['CreditScore'].astype(int)
    df['Approved'] = df['Approved'].astype(int)
    
    df.to_csv(filename, index=False)
    print(f"Dataset saved to '{filename}' with shape {df.shape}")
    print(f"Approval rate: {df['Approved'].mean():.2%}")
    print(f"Number of Age=144: {sum(df['Age'] == 144)}")
    print(f"Extreme Income values (>= 900k): {sum(df['Income'] >= 900000)}")

if __name__ == "__main__":
    generate_synthetic_data()
