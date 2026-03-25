import pandas as pd

df = pd.read_excel("credit_card_fraud_10k.csv.xlsx")
print(df.head())
print(df['Class'].value_counts())

x = df.drop('Class', axis=1)
y = df['Class']

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)
