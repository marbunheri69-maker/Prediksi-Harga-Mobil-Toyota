import pandas as pd

df = pd.read_csv("data/toyota.csv")

print("MODEL")
print(sorted(df["model"].unique()))

print("\nTRANSMISSION")
print(sorted(df["transmission"].unique()))

print("\nFUEL TYPE")
print(sorted(df["fuelType"].unique()))
