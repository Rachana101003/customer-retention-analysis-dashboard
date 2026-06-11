import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Dataset Information
print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Convert TotalCharges to numeric
print("\n===== CONVERTING TOTAL CHARGES =====")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print(df["TotalCharges"].dtype)

# Check missing values after conversion
print("\n===== MISSING VALUES AFTER CONVERSION =====")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

print("\n===== SHAPE AFTER CLEANING =====")
print(df.shape)

# Total Customers
print("\n===== TOTAL CUSTOMERS =====")
print(len(df))

# Churn Rate
churned = len(
    df[df["Churn"] == "Yes"]
)

total_customers = len(df)

churn_rate = (
    churned / total_customers
) * 100

print("\n===== CHURN RATE =====")
print(round(churn_rate, 2))

# Retention Rate
retained = len(
    df[df["Churn"] == "No"]
)

retention_rate = (
    retained / total_customers
) * 100

print("\n===== RETENTION RATE =====")
print(round(retention_rate, 2))

print("\n===== CHURN BY CONTRACT TYPE =====")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

print(contract_churn)

print("\n===== CHURN PERCENTAGE BY CONTRACT =====")

contract_churn_pct = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(round(contract_churn_pct,2))

print("\n===== CUSTOMER SEGMENTATION =====")

df["CustomerSegment"] = pd.qcut(
    df["MonthlyCharges"],
    q=3,
    labels=[
        "Low Value",
        "Medium Value",
        "High Value"
    ]
)

print(
    df["CustomerSegment"].value_counts()
)

print("\n===== CHURN BY CUSTOMER SEGMENT =====")

segment_churn = pd.crosstab(
    df["CustomerSegment"],
    df["Churn"],
    normalize="index"
) * 100

print(round(segment_churn,2))

print("\n===== CUSTOMER LIFETIME VALUE =====")

average_monthly_charge = df["MonthlyCharges"].mean()

average_tenure = df["tenure"].mean()

ltv = average_monthly_charge * average_tenure

print("Average Monthly Charge:", round(average_monthly_charge,2))
print("Average Tenure:", round(average_tenure,2))
print("Estimated LTV:", round(ltv,2))

df.to_csv(
    "cleaned_customer_data.csv",
    index=False
)

print("\nCleaned dataset exported successfully.")

plt.figure(figsize=(6,4))

df["Churn"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")

plt.savefig("reports/churn_distribution.png")
plt.close()

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Contract Type vs Churn")

plt.savefig(
    "reports/contract_vs_churn.png"
)

plt.close()

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="CustomerSegment",
    hue="Churn"
)

plt.title("Customer Segment vs Churn")

plt.savefig(
    "reports/segment_vs_churn.png"
)

plt.close()