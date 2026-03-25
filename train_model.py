import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


df = pd.read_excel("credit_card_fraud_10k.csv.xlsx")
df.columns = df.columns.str.strip()

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

print("After encoding:", df.head())

#separating the features and target variable
x = df.drop('is_fraud', axis=1)
y = df['is_fraud']

#handling imbalance in the dataset using SMOTE
smote = SMOTE()
x_resampled, y_resampled = smote.fit_resample(x, y)
print("Dataset Resampled Successfully✅")

#train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x_resampled, y_resampled, test_size=0.2, random_state=42)

#scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#logistic regression model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

lr_pred = lr_model.predict(x_test)
print("\nLogistic Regression Results")
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

#random forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(x_train, y_train)

rf_pred = rf_model.predict(x_test)
print("\nRandom Forest Results")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
