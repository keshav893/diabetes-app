import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

print("Before cleaning:", data.shape)

# 🔥 Convert ALL object columns automatically
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category').cat.codes

# 🔥 Convert everything to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Remove null values
data = data.dropna()

print("After cleaning:", data.shape)

# Features & target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Safety check
if X.shape[0] == 0:
    print("❌ DATA EMPTY")
    exit()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Small model
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ FINAL MODEL READY (NO ERROR)")