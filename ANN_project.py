# Student Mental Health Risk Classifier
# classification (PHQ-9 risk band)

#Import Libraries 
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay,)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Data Loading
df = pd.read_csv("student_mental_health_hybrid_learning.csv")
print(df.shape)
print(f"Number of observations (rows): {df.shape[0]}")
print(f"Number of attributes (columns): {df.shape[1]}")

# Check Missing Values
print(df.isnull().sum())
# Drop Missing Values
df = df.dropna()
print(df.isnull().sum())


# Standardize Gender Labels 
df["gender"] = df["gender"].str.strip().str.lower()
df["gender"] = df["gender"].replace({
    "m": "Male",
    "male": "Male",
    "f": "Female",
    "female": "Female",
})
df["gender"] = df["gender"].str.title()
print("\nGender value counts:")
print(df["gender"].value_counts())

# Create PHQ-9 Risk Bands (standard clinical cutoffs) 
# 0-4: Minimal | 5-9: Mild | 10-14: Moderate | 15-19: Moderately severe | 20-27: Severe
# Collapsed into 4 classes 
def phq9_to_band(score):
    if score <= 4:
        return "Minimal"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    else:
        return "Moderately Severe/Severe"

df["risk_band"] = df["phq9_score"].apply(phq9_to_band)

print("\nRisk band distribution:")
print(df["risk_band"].value_counts())

# Feature Selection 
# Drop identifier and both raw score + derived label from features
X = df.drop(columns=["student_id", "phq9_score", "risk_band"])
X = pd.get_dummies(X, drop_first=True)  # one-hot encode categorical variables

# Encode target labels to integers
band_order = ["Minimal", "Mild", "Moderate", "Moderately Severe/Severe"]
label_to_int = {label: i for i, label in enumerate(band_order)}
y = df["risk_band"].map(label_to_int)

# Train/Test Split BEFORE scaling 
# test_size=0.5 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=42,
)

# Scale features (fit on train only, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network Classifier
num_classes = len(band_order)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))  # softmax for multi-class classification

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train_scaled,
    y_train_cat,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

loss, accuracy = model.evaluate(X_test_scaled, y_test_cat)
print("\n--- Neural Network ---")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Predictions
pred_probs = model.predict(X_test_scaled)
pred_classes = np.argmax(pred_probs, axis=1)

print("\nClassification Report (Neural Network):")
print(classification_report(y_test, pred_classes, target_names=band_order))

# Confusion Matrix 
# More informative than a single accuracy number: shows WHICH risk bands get confused with each other (e.g. is "Moderate" beingmistaken for "Minimal").
cm = confusion_matrix(y_test, pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=band_order)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix: Predicted vs Actual Risk Band")
plt.tight_layout()
plt.show()

# Training History Plot
plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
