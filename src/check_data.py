import pandas as pd

df = pd.read_csv("data/toyota.csv")

print("===== HEAD =====")
print(df.head())

print("\n===== INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())