# ==============================
# GREEN SPACE MENTAL HEALTH ANALYSIS
# WITH NOVEL CGRI IMPLEMENTATION
# ==============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# ------------------------------
# 1. LOAD DATASET
# ------------------------------

df = pd.read_csv("vit_green_space_realistic_synthetic_dataset.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------
# 2. CORRELATION ANALYSIS
# ------------------------------

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# ------------------------------
# 3. MULTIPLE LINEAR REGRESSION
# ------------------------------

X = df[[
    "green_cover_pct",
    "noise_db",
    "shade_level",
    "seating_count",
    "crowd_density",
    "duration_minutes",
    "exam_period"
]]

y = df["stress_reduction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLinear Regression R2 Score:", r2_score(y_test, y_pred))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance (Linear Regression):")
print(importance)


# ------------------------------
# 4. NOVEL CONTRIBUTION:
# CONTEXT-AWARE GREEN RESILIENCE INDEX (CGRI)
# ------------------------------

# Separate positive and negative factors

positive_features = ["green_cover_pct", "shade_level", "duration_minutes"]
negative_features = ["noise_db", "crowd_density", "exam_period"]

scaler = MinMaxScaler()

# Normalize positive features
df_pos = pd.DataFrame(
    scaler.fit_transform(df[positive_features]),
    columns=positive_features
)

# Normalize negative features
df_neg = pd.DataFrame(
    scaler.fit_transform(df[negative_features]),
    columns=negative_features
)

# Environmental Capacity (EC)
df["EC"] = df_pos.mean(axis=1)

# Stress Pressure (SP)
df["SP"] = df_neg.mean(axis=1)

# CGRI Calculation
df["CGRI"] = df["EC"] - df["SP"]

print("\nSample CGRI Values:")
print(df[["EC", "SP", "CGRI"]].head())


# ------------------------------
# 5. VALIDATE CGRI
# ------------------------------

# Correlation
corr_cgri = df[["CGRI", "stress_reduction"]].corr()
print("\nCorrelation between CGRI and Stress Reduction:")
print(corr_cgri)

# Regression using only CGRI
X_cgri = df[["CGRI"]]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cgri, y, test_size=0.2, random_state=42
)

model_c = LinearRegression()
model_c.fit(X_train_c, y_train_c)

y_pred_c = model_c.predict(X_test_c)

print("\nCGRI Model R2 Score:", r2_score(y_test_c, y_pred_c))


# ------------------------------
# 6. VISUALIZATION
# ------------------------------

plt.figure(figsize=(6,5))
sns.scatterplot(x=df["CGRI"], y=df["stress_reduction"])
plt.title("CGRI vs Stress Reduction")
plt.xlabel("CGRI")
plt.ylabel("Stress Reduction")
plt.show()

# ------------------------------
# 7. RESILIENCE CLASSIFICATION
# ------------------------------

# Create resilience categories
df["Resilience_Level"] = pd.qcut(
    df["CGRI"],
    q=3,
    labels=["Low Resilience", "Moderate Resilience", "High Resilience"]
)

print("\nResilience Distribution:")
print(df["Resilience_Level"].value_counts())

# Average stress reduction per resilience level
print("\nAverage Stress Reduction by Resilience Level:")
print(df.groupby("Resilience_Level")["stress_reduction"].mean())