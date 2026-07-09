# Context
This project was developed as part of a group assignment in the Machine Learning course.

# Background and Motivation
Mental health challenges among university students continue to increase worldwide, 
particularly in hybrid learning environments where students face unique stressors 
combining online and in-person education. Understanding both mental health risk 
factors and their impact on academic performance is critical for institutions 
to offer timely support and intervention.

# Objective
The project aims to:
- Demonstrate the application of multiple machine learning algorithms on real student data
- Compare neural networks and linear regression for different prediction tasks
- Evaluate predictive performance using appropriate metrics
- Support data-driven decision making through interpretable models
- Both analyses use the same 50/50 train-test split (random_state=42) so that
performance is compared under identical data conditions.

# Dataset
student_mental_health_hybrid_learning.csv
- 500 university students in hybrid learning context
- Features include demographics, lifestyle habits, mental health scores, and academic metrics

Key variables:
- Age, Gender, Year of Study, Major
- Daily Sleep Hours, Screen Time, Physical Activity
- PHQ-9 (Depression), GAD-7 (Anxiety), Perceived Stress Scale
- Weekly Study Hours, Class Attendance, Online Learning Satisfaction
- Financial Stress, Part-time Job Status
- CGPA (Cumulative GPA)

# Methodology

## Analysis 1: PHQ-9 Risk Classification

-Algorithm: Artificial Neural Network (Keras Sequential,64→32→4 softmax)
-Target: PHQ-9 score, converted into 4 clinical risk bands (Minimal / Mild /
Moderate / Moderately Severe-Severe) rather than predicted as a raw number
-Rationale: PHQ-9 is a self-reported screening measure. Framing
it as risk-band classification mirrors how PHQ-9 is actually used in practice and allows evaluation via confusion matrix, showing which
risk levels get confused with each other.
-Preprocessing: rows with missing values dropped, gender labels standardized,
categorical variables one-hot encoded, features scaled 

## Analysis 2: CGPA Prediction
- Algorithm: Multiple Linear Regression
- Target: CGPA (academic performance)
- Rationale: Linear regression provides interpretable coefficients for academic advising
- Preprocessing: Box-Cox transformation (square root), model selection via AIC/BIC/Mallows' Cp

# Tools and Technologies
- Python
- pandas, NumPy
- scikit-learn (preprocessing, train-test split,metrics)
- TensorFlow/Keras(neural network)
- statsmodels (linear regression, AIC, BIC)
- scipy (Box-Cox transformation)
- matplotlib (confusion matrix, training curves)

# Results

## PHQ-9 Risk Classification (ANN)

MetricValueTest Accuracy: 59.3%
Test Loss: 1.43
Classification report (test set, n=226):

Confusion matrix takeaway: the model is most reliable at the extremes,
"Minimal" (0.73 F1) and "Moderately Severe/Severe" (0.65 F1) are identified
with reasonable confidence. The middle bands, "Mild" and "Moderate," are
where the model struggles most (F1 of 0.46 and 0.44), and this is also
where most misclassifications land — adjacent-band confusion (e.g., Mild
predicted as Moderate or vice versa) rather than the far more serious
Minimal-vs-Severe type of error. This is the more forgivable failure mode
for a screening tool: the model rarely mistakes a low-risk student for a
high-risk one.

Overfitting note: training accuracy reached 100% by the later epochs
while validation accuracy plateaued around 64–67%, and validation loss
climbed steadily after roughly epoch 20–30 while training loss kept
falling. This is a classic overfitting pattern, expected given the modest
dataset size (500 students, only ~250 in the training split after the
50/50 divide). Worth mentioning as a limitation: with more data, or
regularization (dropout, L2, early stopping), the model would likely
generalize better. This is also a reasonable talking point if asked why a
simpler model might be competitive here — a small, noisy dataset limits
how much a more complex model can actually exploit.


## CGPA Prediction (Linear Regression)
- Final model: 6 significant predictors from 16 original variables
- R² = 0.428, Adjusted R² = 0.408 
- Key predictors: weekly study hours, class attendance, sleep, anxiety (GAD-7), part-time job

# Conclusion
This comparative analysis illustrates a fundamental machine learning principle: 
algorithm selection should match both the data characteristics and the decision context. 
Neural networks excel at capturing complex patterns in mental health data, while linear 
regression offers actionable, interpretable insights for academic performance prediction.
