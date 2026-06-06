import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual theme
sns.set_theme(style="whitegrid")
print("🏬 Starting Task 4: Real-world Retail Data Project...")

# 1. GENERATE A DOMAIN-SPECIFIC RETAIL DATASET
np.random.seed(42)
n_records = 200

months = np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], size=n_records)
categories = np.random.choice(['Electronics', 'Clothing', 'Home Decor', 'Groceries'], size=n_records, p=[0.25, 0.35, 0.15, 0.25])
prices = np.random.uniform(10.0, 500.0, size=n_records).round(2)
quantities = np.random.randint(1, 5, size=n_records)

df = pd.DataFrame({
    'Month': months,
    'Category': categories,
    'Price': prices,
    'Quantity': quantities
})

# Feature Engineering: Calculate Total Revenue per transaction
df['Total_Revenue'] = df['Price'] * df['Quantity']

# 2. PERFORM END-TO-END DATA ANALYSIS (Aggregations)
print("\n📋 --- Total Revenue Generated per Product Category ---")
category_summary = df.groupby('Category')['Total_Revenue'].sum().reset_index()
print(category_summary.round(2))

print("\n📈 --- Monthly Sales Performance Summary ---")
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
monthly_summary = df.groupby('Month', observed=False)['Total_Revenue'].sum().reset_index()
print(monthly_summary.round(2))

# 3. PRESENT FINDINGS WITH VISUALIZATIONS
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Revenue Breakdown by Category
sns.barplot(data=category_summary, x='Category', y='Total_Revenue', hue='Category', palette='Set2', legend=False, ax=axes[0])
axes[0].set_title('1. Total Revenue Breakdown by Category')
axes[0].set_xlabel('Product Category')
axes[0].set_ylabel('Total Revenue ($)')

# Plot 2: Monthly Sales Performance Trend Line
sns.lineplot(data=monthly_summary, x='Month', y='Total_Revenue', marker='o', color='purple', linewidth=2.5, ax=axes[1])
axes[1].set_title('2. 2026 First-Half Monthly Sales Trend')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Total Revenue ($)')

plt.tight_layout()

# Save the real-world domain visual report
plt.savefig('retail_sales_report.png')
print("\n💾 Visual Report successfully saved as 'retail_sales_report.png'")