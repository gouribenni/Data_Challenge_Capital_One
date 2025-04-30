import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd


def plot_missing_values(df):
    # Heatmap showing nulls in white
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.show()


def plot_outliers(df, column_list, outlier_mask=None, method_name="Z-Score"):
    for column in column_list:
        plt.figure(figsize=(14, 5))

        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot of {column} ({method_name} Outliers)")

        # KDE plot with outliers
        plt.subplot(1, 2, 2)
        sns.kdeplot(df[column], label="All Data", fill=True)
        if outlier_mask is not None:
            sns.kdeplot(df.loc[outlier_mask, column], label="Outliers", color="red", fill=True)
        plt.title(f"KDE of {column} with Outliers Highlighted ({method_name})")
        plt.legend()

        plt.tight_layout()
        plt.show()


def batch_plot_outliers(df, column_list, method='zscore', z_thresh=3, iqr_factor=1.5):
    for col in column_list:
        print(f"\nAnalyzing Outliers for: {col} using {method.upper()}")
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores > z_thresh
        elif method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[col] < (Q1 - iqr_factor * IQR)) | (df[col] > (Q3 + iqr_factor * IQR))
        else:
            raise ValueError("Method must be either 'zscore' or 'iqr'")

        plot_outliers(df, [col], outlier_mask=mask, method_name=method.upper())


def outlier_summary_report(df, column_list, method='zscore', z_thresh=3, iqr_factor=1.5):
    report = []
    for col in column_list:
        data = df[col].dropna()
        skewness = stats.skew(data)
        kurt = stats.kurtosis(data)
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outlier_count = (z_scores > z_thresh).sum()
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (data < (Q1 - iqr_factor * IQR)) | (data > (Q3 + iqr_factor * IQR))
            outlier_count = outlier_mask.sum()
        else:
            raise ValueError("Method must be either 'zscore' or 'iqr'")

        report.append({
            "Column": col,
            "Skewness": round(skewness, 2),
            "Kurtosis": round(kurt, 2),
            "Outlier Count": outlier_count
        })

    return pd.DataFrame(report)


def cap_outliers(df, column, k=3):
    df = df.copy()
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    lower = median - k * mad
    upper = median + k * mad
    df[column] = df[column].clip(lower, upper)
    return df

def cap_and_report_outliers(df, column, k=3):
    original_values = df[column].copy()

    median = original_values.median()
    mad = np.median(np.abs(original_values - median))
    lower = median - k * mad
    upper = median + k * mad

    capped_values = original_values.clip(lower, upper)
    capped_rows = (np.abs(original_values - capped_values) > 1e-8).sum()

    print(f"\nOutlier Capping Summary for '{column}':")
    print(f"  Median: {median:.2f}")
    print(f"  MAD: {mad:.2f}")
    print(f"  Boundaries: [{lower:.2f}, {upper:.2f}]")
    print(f"  Rows capped: {capped_rows} ({(capped_rows / len(df)) * 100:.2f}%)")

    df[column] = capped_values
    return df



def print_outlier_summary(df, column, k):
    original_values = df[column].copy()
    median = original_values.median()
    mad = np.median(np.abs(original_values - median))
    lower = median - k * mad
    upper = median + k * mad

    capped_values = original_values.clip(lower, upper)

    # Use numerical tolerance for float comparison
    capped_rows = (np.abs(original_values - capped_values) > 1e-8).sum()

    print(f"Outlier Capping Summary for '{column}':")
    print(f"  Median: {median:.2f}")
    print(f"  MAD: {mad:.2f}")
    print(f"  Boundaries: [{lower:.2f}, {upper:.2f}]")
    print(f"  Rows capped: {capped_rows} ({(capped_rows / len(df)) * 100:.2f}%)")

    # Update the dataframe
    df[column] = capped_values

