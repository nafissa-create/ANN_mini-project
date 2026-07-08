#needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("student_mental_health_hybrid_learning.csv")

# Drop student_ID because it is not a predictor
df = df.drop(columns=['student_id'])

# Clean gender (fix "M", "male", "Male")
df['gender'] = df['gender'].str.strip().str.lower()
gender_map = {
    'male': 'male', 'm': 'male',
    'female': 'female', 'f': 'female',
    'non-binary': 'non-binary',
    'prefer not to say': 'prefer not to say'
}
df['gender'] = df['gender'].map(gender_map)

# Drop rows with missing target (CGPA)
df = df.dropna(subset=['cgpa'])

# Fill missing numeric values with median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Encode all text columns to numbers
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# split the data before any modelling
train, test = train_test_split(df, test_size=0.50, random_state=42)
print(f"Training set: {len(train)} students")
print(f"Testing set:  {len(test)} students")

# Training data only
X_train = train.drop('cgpa', axis=1)
y_train = train['cgpa']
X_train_const = sm.add_constant(X_train)

# Test data (held out completely)
X_test = test.drop('cgpa', axis=1)
y_test = test['cgpa']

# data summary
print(f"\nShape: {X_train.shape}")
print(f"Features: {list(X_train.columns)}")
print(f"\nCGPA stats (training):\n{y_train.describe()}")

# fit a full multiple linear regression model on TRAINING data only
model_full = sm.OLS(y_train, X_train_const).fit()
print(model_full.summary())

residuals = model_full.resid

# Shapiro-Wilk test for normality
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\n{'='*60}")
print("RESIDUAL NORMALITY TEST (Training Data)")
print(f"{'='*60}")
print(f"Shapiro-Wilk Statistic: {shapiro_stat:.4f}")
print(f"Shapiro-Wilk p-value:   {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print(" Residuals are not normal. Box-Cox transformation is needed.")
else:
    print(" Residuals look normal.")
 

# Box-cox transformation on TRAINING target only
# Box-Cox requires strictly positive values

# Get optimal lambda from Box-Cox
_, lmbda_optimal = stats.boxcox(y_train)
print(f"\nOptimal Box-Cox lambda: {lmbda_optimal:.4f}")

# Round to nearest interpretable lambda
supported = [2, 1, 0.5, 0, -0.5, -2]
lmbda = min(supported, key=lambda x: abs(x - lmbda_optimal))
print(f"Rounded lambda: {lmbda}")

# Apply transformation
if lmbda == 1:
    y_train_best = y_train.copy()
    transform_name = "No transformation"
elif lmbda == 0.5:
    y_train_best = np.sqrt(y_train)
    transform_name = "Square root"
elif lmbda == 0:
    y_train_best = np.log(y_train)
    transform_name = "Natural log"
elif lmbda == -0.5:
    y_train_best = 1 / np.sqrt(y_train)
    transform_name = "Inverse square root"
elif lmbda == 2:
    y_train_best = y_train ** 2
    transform_name = "Square"
elif lmbda == -2:
    y_train_best = 1 / (y_train ** 2)
    transform_name = "Inverse square"

print(f"Transformation needed is the: {transform_name}")

# Fit the model 
best_model = sm.OLS(y_train_best, X_train_const).fit()
print("\n--- TRANSFORMED MODEL (Training Data) ---")
print(best_model.summary())

# model selection on training data
# Identify significant predictors from the full model (p < 0.05)
# We keep major_category too since p = 0.048 is borderline
significant_variables = [
    'major_category',
    'daily_sleep_hours',
    'gad7_score',
    'weekly_study_hours',
    'class_attendance_percent',
    'has_part_time_job'
]

# Build reduced model on TRAINING data
X_train_reduced = X_train[significant_variables]
X_train_reduced_const = sm.add_constant(X_train_reduced)
model_reduced = sm.OLS(y_train_best, X_train_reduced_const).fit()

# Mallows' Cp 
def mallows_cp(model_subset, model_full, n):
    p_subset = model_subset.params.shape[0]
    sse_subset = sum(model_subset.resid ** 2)
    mse_full = sum(model_full.resid ** 2) / model_full.df_resid
    cp = sse_subset / mse_full - (n - 2 * p_subset)
    return cp

n = len(y_train_best)
cp_full = mallows_cp(best_model, best_model, n)
cp_reduced = mallows_cp(model_reduced, best_model, n)

# Compare
print("\n" + "="*70)
print("MODEL SELECTION TABLE")
print("="*70)
print(f"{'Model':<20} {'R²':>8} {'Adj R²':>8} {'AIC':>10} {'BIC':>10} {'Cp':>10}")
print("-"*70)
print(f"{'Full (16 vars)':<20} {best_model.rsquared:>8.4f} {best_model.rsquared_adj:>8.4f} {best_model.aic:>10.2f} {best_model.bic:>10.2f} {cp_full:>10.2f}")
print(f"{'Reduced (6 vars)':<20} {model_reduced.rsquared:>8.4f} {model_reduced.rsquared_adj:>8.4f} {model_reduced.aic:>10.2f} {model_reduced.bic:>10.2f} {cp_reduced:>10.2f}")

# Pick the best
if model_reduced.aic < best_model.aic:
    print(f"\n→ Reduced model is the best (lower AIC by {best_model.aic - model_reduced.aic:.2f}).")
    final_model = model_reduced
    final_vars = significant_variables
    final_name = "Reduced"
else:
    print(f"\n→ Full model is the best")
    final_model = best_model
    final_vars = list(X_train.columns)
    final_name = "Full"

print(final_model.summary())

# predict on test data

# Use same variables as final model
X_test_final = X_test[final_vars]
X_test_final_const = sm.add_constant(X_test_final)

# Check if we used transformation
if lmbda == 0.5:  # square root was used
    y_pred_sqrt = final_model.predict(X_test_final_const)
    y_pred = y_pred_sqrt ** 2
else:
    y_pred = final_model.predict(X_test_final_const)

# Evaluate
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred)**2))

print(f"\n{'='*60}")
print("TEST SET PERFORMANCE (Unseen 50% of data)")
print(f"{'='*60}")
print(f"Mean Absolute Error: {mae:.3f} GPA points")
print(f"Root Mean Squared Error: {rmse:.3f} GPA points")

# Show first 10 predictions
print("\n--- First 10 Predictions vs Actual ---")
results = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': np.round(y_pred.values[:10], 2)
})
print(results.to_string(index=False))

# Manual prediction example, allow the user to enter their own data

print("\n" + "="*60)
print("Predict your own CGPA😁")
print("="*60)
print("Enter the following information:\n")

try:
    sleep = float(input("Daily sleep hours: "))
    gad7 = float(input("GAD-7 anxiety score (0-21): "))
    study = float(input("Weekly study hours: "))
    attend = float(input("Class attendance %: "))
    job = int(input("Has part-time job? (1=Yes, 0=No): "))
    major = int(input("Major category (0=Arts, 1=Business, 2=Health Sciences, 3=Humanities, 4=Law, 5=Social Sciences, 6=STEM): "))

    # Build input from what final model actually uses
    user_input = {}
    if 'daily_sleep_hours' in final_vars:
        user_input['daily_sleep_hours'] = sleep
    if 'gad7_score' in final_vars:
        user_input['gad7_score'] = gad7
    if 'weekly_study_hours' in final_vars:
        user_input['weekly_study_hours'] = study
    if 'class_attendance_percent' in final_vars:
        user_input['class_attendance_percent'] = attend
    if 'has_part_time_job' in final_vars:
        user_input['has_part_time_job'] = job
    if 'major_category' in final_vars:
        user_input['major_category'] = major

    X_user = pd.DataFrame([user_input])
    X_user_const = sm.add_constant(X_user, has_constant='add')
    pred = final_model.predict(X_user_const)[0]

    if lmbda == 0.5:
        pred = pred ** 2

    print(f"\n→ Predicted CGPA: {pred:.2f}")

except ValueError:
    print("Invalid input. Please enter numbers only.")
except KeyboardInterrupt:
    print("\nExiting.")
