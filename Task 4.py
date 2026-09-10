# =========================================================
# YUVA INTERNSHIP - DATA SCIENCE
# Week 4 Task: Supervised Learning Model Implementation
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("--- Starting Week 4 Task: Supervised Learning (Classification) ---")

# 1. Load Dataset
df = sns.load_dataset('titanic')

# 2. Data Preprocessing & Feature Engineering
print("Preprocessing Data...")
# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)

# Drop irrelevant columns (deck dropped due to high nulls)
df.drop(['deck', 'alive', 'who', 'adult_male', 'class', 'embark_town'], axis=1, inplace=True, errors='ignore')

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Define Features (X) and Target (y)
X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Implementation
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Fit the model
model.fit(X_train, y_train)

# 5. Model Evaluation
print("\nEvaluating Model...")
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Visualization 1: Confusion Matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Deceased', 'Survived'], yticklabels=['Deceased', 'Survived'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('Confusion_Matrix.png', bbox_inches='tight')
plt.show()

# Visualization 2: Feature Importance
plt.figure(figsize=(8, 5))
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis')
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.savefig('Feature_Importance.png', bbox_inches='tight')
plt.show()

print("\n--- Model Implementation Completed. Graphs saved as PNG files! ---")