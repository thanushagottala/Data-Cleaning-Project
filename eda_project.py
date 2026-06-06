import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual theme
sns.set_theme(style="whitegrid")
print("📊 Starting Exploratory Data Analysis (EDA)...")

# 1. GENERATE A MULTI-VARIABLE DATASET
# Simulating an employee performance and salary dataset
np.random.seed(101)
n_samples = 150
data = {
    'Experience_Years': np.random.randint(1, 15, size=n_samples),
    'Education_Level': np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.3, 0.2]), # 1:Bachelors, 2:Masters, 3:PhD
    'Projects_Completed': np.random.randint(2, 10, size=n_samples)
}
df = pd.DataFrame(data)

# Add dependent continuous variables with correlation patterns
df['Salary_USD'] = 30000 + (df['Experience_Years'] * 4500) + (df['Education_Level'] * 8000) + np.random.normal(0, 5000, n_samples)
df['Performance_Score'] = 50 + (df['Projects_Completed'] * 4) + (df['Experience_Years'] * 1.5) + np.random.normal(0, 5, n_samples)

# 2. STATISTICAL SUMMARIES
print("\n📋 --- Descriptive Statistical Summary ---")
print(df.describe().round(2))

# 3. IDENTIFY CORRELATIONS
print("\n🔗 --- Correlation Matrix (Key Influencing Factors) ---")
correlation_matrix = df.corr()
print(correlation_matrix.round(2))

# 4. STRUCTURED VISUAL DATA EXPLORATION REPORT
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Correlation Heatmap to show influencing factors
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0], cbar=True)
axes[0].set_title('1. Correlation Matrix Heatmap')

# Plot 2: Scatter plot showcasing Key Trend (Experience vs Salary)
sns.regplot(data=df, x='Experience_Years', y='Salary_USD', scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=axes[1])
axes[1].set_title('2. Salary Trend vs Experience Years')
axes[1].set_xlabel('Years of Experience')
axes[1].set_ylabel('Salary ($)')

plt.tight_layout()

# Save the structured EDA report plot
plt.savefig('eda_report.png')
print("\n💾 Structured visual report successfully saved as 'eda_report.png'")