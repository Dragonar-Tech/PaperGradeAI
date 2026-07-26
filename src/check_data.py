import pandas as pd

df = pd.read_csv("data/paper_grade_data.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nOff-Spec Distribution:")
print(df["Off_Spec"].value_counts())
print("\nOff-Spec Rate:", round(df["Off_Spec"].mean(), 3))
print("\nMissing values:")
print(df.isna().sum().sum(), "total missing cells")
