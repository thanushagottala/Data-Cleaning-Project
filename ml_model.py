import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Set style for visualizations
sns.set_theme(style="whitegrid")

print("🔄 Initializing Machine Learning Process...")

# 1. GENERATE DATASET FOR MACHINE LEARNING
np.random.seed(42)
ages = np.random.randint(18, 65, size=100)
salaries = np.random.randint(20000, 150000, size=100)
purchased = np.where((ages * salaries > 2000000), 1, 0)

df = pd.DataFrame({'Age': ages, 'Salary': salaries, 'Purchased': purchased})

# 2. SPLIT DATA INTO TRAINING & TESTING SETS
X = df[['Age', 'Salary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. APPLY ALGORITHM (Decision Tree)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. TRAIN AND TEST MODELS FOR ACCURACY
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Training Successful!")
print(f"🎯 Prediction Accuracy: {accuracy * 100:.1f}%\n")

# 5. VISUALIZE PERFORMANCE (CONFUSION MATRIX)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Predicted: No', 'Predicted: Yes'], 
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title('Task 2: Decision Tree Confusion Matrix')
plt.tight_layout()

# Save the visual report image file automatically
plt.savefig('ml_confusion_matrix.png')
print("💾 Visual Report saved as 'ml_confusion_matrix.png'")