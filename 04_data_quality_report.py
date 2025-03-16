import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('scraped_datasets/fb_comments/SAMPLE_ALL_COMMENTS_PREPROCESSED_DATASET.csv')

# Data types
def categorize_data_types(df):
    nominal = []
    ordinal = []
    continuous = []
    binary = []

    for col in df.columns:
        unique_vals = df[col].nunique()
        dtype = df[col].dtype

        if dtype == 'object':
            nominal.append(col)
        elif dtype in ['int64', 'float64']:
            if unique_vals == 2:
                binary.append(col)
            elif unique_vals <= 10:
                ordinal.append(col)
            else:
                continuous.append(col)

    print("Nominal Variables:", nominal)
    print("Ordinal Variables:", ordinal)
    print("Continuous Variables:", continuous)
    print("Binary Variables:", binary)

categorize_data_types(data)

# DATA QUALITY REPORT
def data_quality_report(df):
    print("DATA QUALITY REPORT \n")

    missing_vals = df.isnull().sum()
    print("Missing Values per Column:")
    print(missing_vals[missing_vals > 0], "\n")

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicate_count}\n")

    print("Unique Values per Column:")
    print(df.nunique(), "\n")

    print("Potential Outliers:")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).sum()
    print(outliers[outliers > 0], "\n")

    print("Checking Data Types for Consistency:")
    print(df.dtypes, "\n")

data_quality_report(data)


# Non-null and null values for each column
column_stats = {
    "Column": [],
    "Non-Null Count": [],
    "Null Count": []
}

for col in data.columns:
    column_stats["Column"].append(col)
    column_stats["Non-Null Count"].append(data[col].notnull().sum())
    column_stats["Null Count"].append(data[col].isnull().sum())

# Convert to DataFrame and display results
stats_df = pd.DataFrame(column_stats)
print(stats_df)

