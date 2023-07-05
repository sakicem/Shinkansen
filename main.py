from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier
from asd import get_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

df_train, df_test = get_data()

# Separate features and target variable
X_train = df_train.drop("Overall_Experience", axis=1)
y_train = df_train["Overall_Experience"]
X_test = df_test.drop("Overall_Experience", axis=1)
y_test = df_test["Overall_Experience"]


# Identify categorical columns
categorical_columns = X_train.select_dtypes(include=["object"]).columns

# Select features for training
features = [
    'Seat_Comfort', 'Seat_Class', 'Arrival_Time_Convenient', 'Catering', 'Platform_Location',
    'Onboard_Wifi_Service', 'Onboard_Entertainment', 'Online_Support', 'Ease_of_Online_Booking',
    'Onboard_Service', 'Legroom', 'Baggage_Handling', 'CheckIn_Service', 'Cleanliness', 'Online_Boarding'
]


# Separate features and target
X_train = df_train.drop('Overall_Experience', axis=1)
y_train = df_train['Overall_Experience']

# Apply ordinal encoding to categorical features
encoder = OrdinalEncoder()
categorical_cols = ['Seat_Class', 'Arrival_Time_Convenient', 'Catering', 'Platform_Location', 'Onboard_Wifi_Service', 'Onboard_Entertainment', 'Online_Support', 'Ease_of_Online_Booking', 'Onboard_Service', 'Legroom', 'Baggage_Handling', 'CheckIn_Service', 'Cleanliness', 'Online_Boarding']

X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Preprocess the test data using the same encoder
X_test = df_test[X_train.columns]  # Use only the same columns as the training data
X_test.loc[:, categorical_cols] = encoder.transform(X_test.loc[:, categorical_cols])

# Make predictions
y_pred = clf.predict(X_test)

# Save predictions to submission file
submission_df = pd.DataFrame({'ID': df_test['ID'], 'Overall_Experience': y_pred})
submission_df.to_csv('submission.csv', index=False)