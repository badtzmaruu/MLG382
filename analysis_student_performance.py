import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
df = pd.read_csv("data/student_performance_data.csv")

# ----- 1. Handle Missing Data -----
print("Missing Values:\n", df.isnull().sum())

# Handling missing values by filling with median (for numerical) or mode (for categorical)
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numerical columns with median
for col in df.select_dtypes(include=['object']).columns:  # Fill categorical columns with mode
    df[col].fillna(df[col].mode()[0], inplace=True)

# ----- 2. Univariate Analysis -----
# Numerical columns
numerical_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA', 'GradeClass']
# Categorical columns
categorical_cols = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 
                    'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']

# Summary statistics for numerical columns
print("\nNumerical Summary:\n", df[numerical_cols].describe())

# Histograms for numerical features
for col in numerical_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Count plots for categorical features
for col in categorical_cols:
    sns.countplot(x=col, data=df)
    plt.title(f'Count Plot for {col}')
    plt.show()

# ----- 3. Bivariate Analysis -----
# Correlation Heatmap (numerical columns)
plt.figure(figsize=(10, 8))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Boxplots for categorical vs numerical (GPA, StudyTime, etc.)
for col in categorical_cols:
    sns.boxplot(x=col, y='GPA', data=df)
    plt.title(f'GPA by {col}')
    plt.show()

# Scatter plots: Numerical vs GPA
for col in numerical_cols:
    if col != 'GPA':
        sns.scatterplot(x=col, y='GPA', data=df)
        plt.title(f'GPA vs {col}')
        plt.show()

# ----- 4. Outlier Detection and Treatment -----
# Using Z-scores to detect outliers
z_scores = stats.zscore(df[numerical_cols])
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df_cleaned = df[filtered_entries]

print("\nAfter outlier removal:", df_cleaned.shape)

# Optionally, visualize outliers (can be included or omitted)
for col in numerical_cols:
    sns.boxplot(x=df_cleaned[col])
    plt.title(f'Boxplot for {col} (Outliers Removed)')
    plt.show()
