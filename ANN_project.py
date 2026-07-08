#Import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#Data loading
df = pd.read_csv("student_mental_health_hybrid_learning.csv")
#Dataset Dimensions
print("\nDataset Shape:")
print(df.shape)
print(f"\nNumber of observations (rows): {df.shape[0]}")
print(f"Number of attributes (columns): {df.shape[1]}")
#Check Missing Values
print(df.isnull().sum())
#Remove rows containing missing values
df = df.dropna()
print(df.isnull().sum())
#Standardize Gender Labels
#Remove spaces
df["gender"] = df["gender"].str.strip()
# Convert all labels to lowercase
df["gender"] = df["gender"].str.lower()
# Replace inconsistent labels
df["gender"] = df["gender"].replace({
    "m": "Male",
    "male": "Male",
    "f": "Female",
    "female": "Female"
})
# Capitalize any remaining values
df["gender"] = df["gender"].str.title()
print(df["gender"].value_counts())
#Feature Selection
X = df.drop(columns=["student_id", "phq9_score"])
X = pd.get_dummies(X, drop_first=True) #one-hot encoding for categorical variables
y = df["phq9_score"] #our target variable
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42
)
# Build the ANN Model

model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

# Hidden Layer 2
model.add(Dense(32, activation='relu'))

# Output Layer
model.add(Dense(1))
# Compile the Model

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# Train the ANN

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
loss, mae = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Mean Absolute Error:", mae)

# Predict PHQ-9 Scores


predictions = model.predict(X_test)

print(predictions[:10])
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

# Actual vs Predicted Values


plt.figure(figsize=(7,7))

plt.scatter(y_test, predictions)

plt.xlabel("Actual PHQ-9 Score")
plt.ylabel("Predicted PHQ-9 Score")
plt.title("Actual vs Predicted PHQ-9 Scores")

plt.show()
