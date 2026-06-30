import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Convert to Pandas DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add species column
df['species'] = pd.Categorical.from_codes(
    iris.target,
    iris.target_names
)

"""
Basic Data Cleaning
"""

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Check for duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()

# -------------------------
# Summary Statistics
# -------------------------

print("\nSummary Statistics:")
print(df.describe())

"""Visualization 1:
 Species Distribution"""

plt.figure(figsize=(6,4))
df['species'].value_counts().plot(kind='bar')
plt.title('Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

"""
Visualization 2:
Sepal Length vs Sepal Width"""

plt.figure(figsize=(6,4))

for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(
        subset['sepal length (cm)'],
        subset['sepal width (cm)'],
        label=species
    )

plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()

# -------------------------
# Visualization 3:
# Petal Length Distribution
# -------------------------

plt.figure(figsize=(6,4))
plt.hist(df['petal length (cm)'], bins=15)
plt.title('Petal Length Distribution')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()