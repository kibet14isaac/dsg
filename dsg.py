import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Initial Exploration ---
print("--- 1. Data Loading and Initial Exploration ---")
# Load the dataset
try:
    df = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please make sure the file is in the same directory as the script.")
    # Create a dummy dataframe for demonstration if file not found
    data = {
        'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10, 2, 8],
        'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125],
        'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96],
        'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0],
        'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232],
        'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54],
        'Outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    print("Using a dummy dataset for demonstration.")


print("\nDataset Head:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nMissing values (0s interpreted as missing in some columns):")
# Columns where 0s represent missing or invalid data
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    print(f"{col}: {(df[col] == 0).sum()} zeros")

# --- 2. Preprocess Data ---
print("\n--- 2. Preprocess Data ---")

# Replace 0s with NaN for imputation in relevant columns
for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)

# Impute missing values with the median (a robust choice for skewed data)
for col in cols_with_zero:
    df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y ensures that the proportion of target variable is the same in both train and test sets

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled successfully.")

# --- 3. Train Model ---
print("\n--- 3. Train Model ---")

# --- Model 1: Logistic Regression ---
print("\n--- Training Logistic Regression Model ---")
lr_model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets
lr_model.fit(X_train_scaled, y_train)
print("Logistic Regression model trained.")

# --- Model 2: Random Forest Classifier ---
print("\n--- Training Random Forest Classifier Model ---")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100) # n_estimators is the number of trees
rf_model.fit(X_train_scaled, y_train)
print("Random Forest Classifier model trained.")

# --- 4. Evaluate Models ---
print("\n--- 4. Evaluate Models ---")

# Evaluate Logistic Regression
print("\n--- Logistic Regression Evaluation ---")
y_pred_lr = lr_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# Plot Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['No Diabetes', 'Diabetes'])
disp_lr.plot(cmap=plt.cm.Blues)
plt.title('Logistic Regression Confusion Matrix')
plt.grid(False) # Turn off grid for cleaner look
plt.show()

# Evaluate Random Forest
print("\n--- Random Forest Classifier Evaluation ---")
y_pred_rf = rf_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Plot Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No Diabetes', 'Diabetes'])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title('Random Forest Classifier Confusion Matrix')
plt.grid(False) # Turn off grid for cleaner look
plt.show()

# --- Feature Importance (for Random Forest) ---
print("\n--- Feature Importance (from Random Forest) ---")
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('Feature Importance from Random Forest Classifier')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()