import numpy as np
import pandas as pd
import sklearn.metrics as ac
import sklearn.model_selection as kfx

data= pd.read_csv("teen_mental_health.csv")
print(data.head())
print("Shape:", data.shape)
print(data.isnull().sum())
