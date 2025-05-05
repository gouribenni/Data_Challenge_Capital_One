import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd

def plot_missing_values(df):
    """
    Plots a heatmap showing the location of missing values in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to visualize missing values.
    """
    # Heatmap where nulls are highlighted (white)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.show()


def plot_outliers(df, column_list, outlier_mask=None, method_name="Z-Score"):
    """
    Plots boxplots and KDE plots for each column to visualize outliers.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_list (list): List of column names to plot.
    - outlier_mask (Series or None): Boolean mask indicating outlier rows.
    - method_name (str): Name of the outlier detection method (for titles).
    """
    for column in column_list:
        plt.figure(figsize=(14, 5))

        # Left subplot: Boxplot for visualizing distribution and outliers
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot of {column} ({method_name} Outliers)")

        # Right subplot: KDE plot showing all data and outliers separately
        plt.subplot(1, 2, 2)
        sns.kdeplot(df[column], label="All Data", fill=True)
        if outlier_mask is not None:
            sns.kdeplot(df.loc[outlier_mask, column], label="Outliers", color="red", fill=True)
        plt.title(f"KDE of {column} with Outliers Highlighted ({method_name})")
        plt.legend()

        plt.tight_layout()
        plt.show()


def batch_plot_outliers(df, column_list, method='zscore', z_thresh=3, iqr_factor=1.5):
    """
    Detects and visualizes outliers for a batch of numeric columns using Z-score or IQR method.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_list (list): List of numeric columns to analyze.
    - method (str): Method for outlier detection ('zscore' or 'iqr').
    - z_thresh (float): Z-score threshold for outlier detection.
    - iqr_factor (float): Multiplier for IQR method.
    """
    for col in column_list:
        print(f"\nAnalyzing Outliers for: {col} using {method.upper()}")

        if method == 'zscore':
            # Compute Z-scores and flag outliers
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores > z_thresh
        elif method == 'iqr':
            # Compute IQR and flag outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[col] < (Q1 - iqr_factor * IQR)) | (df[col] > (Q3 + iqr_factor * IQR))
        else:
            raise ValueError("Method must be either 'zscore' or 'iqr'")

        # Plot results for each column
        plot_outliers(df, [col], outlier_mask=mask, method_name=method.upper())


def outlier_summary_report(df, column_list, method='zscore', z_thresh=3, iqr_factor=1.5):
    """
    Generates a summary report of skewness, kurtosis, and outlier count for each column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_list (list): List of numeric columns to analyze.
    - method (str): Outlier detection method ('zscore' or 'iqr').
    - z_thresh (float): Threshold for Z-score method.
    - iqr_factor (float): IQR multiplier for outlier bounds.

    Returns:
    - pd.DataFrame: Summary table with column name, skewness, kurtosis, and outlier count.
    """
    report = []
    for col in column_list:
        data = df[col].dropna()

        # Statistical shape of the distribution
        skewness = stats.skew(data)
        kurt = stats.kurtosis(data)

        # Outlier count by selected method
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

        # Append to final report
        report.append({
            "Column": col,
            "Skewness": round(skewness, 2),
            "Kurtosis": round(kurt, 2),
            "Outlier Count": outlier_count
        })

    return pd.DataFrame(report)


    def cap_outliers(df, column, k=3):
    """
    Caps outliers in a column using the Median Absolute Deviation (MAD) method.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): Name of the numeric column to cap.
    - k (float): Scaling factor for the MAD (default is 3).

    Returns:
    - pd.DataFrame: A copy of the DataFrame with outliers capped in the specified column.
    """
    df = df.copy()

    # Compute median and MAD
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))

    # Define lower and upper bounds
    lower = median - k * mad
    upper = median + k * mad

    # Cap values outside the range
    df[column] = df[column].clip(lower, upper)
    return df


def cap_and_report_outliers(df, column, k=3):
    """
    Caps outliers in a column using MAD and prints a detailed summary report.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): Column name to process.
    - k (float): Scaling factor for MAD (default = 3).

    Returns:
    - pd.DataFrame: Updated DataFrame with capped values in the given column.
    """
    original_values = df[column].copy()

    # Calculate median and MAD
    median = original_values.median()
    mad = np.median(np.abs(original_values - median))

    # Define MAD-based boundaries
    lower = median - k * mad
    upper = median + k * mad

    # Cap values
    capped_values = original_values.clip(lower, upper)

    # Count how many rows changed
    capped_rows = (np.abs(original_values - capped_values) > 1e-8).sum()

    # Print summary
    print(f"\nOutlier Capping Summary for '{column}':")
    print(f"  Median: {median:.2f}")
    print(f"  MAD: {mad:.2f}")
    print(f"  Boundaries: [{lower:.2f}, {upper:.2f}]")
    print(f"  Rows capped: {capped_rows} ({(capped_rows / len(df)) * 100:.2f}%)")

    # Replace column with capped values
    df[column] = capped_values
    return df


def print_outlier_summary(df, column, k):
    """
    Prints outlier summary for a column without modifying the DataFrame in place.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): Column to evaluate for outliers.
    - k (float): Scaling factor for MAD (default = 3).
    """
    original_values = df[column].copy()

    # Compute median and MAD
    median = original_values.median()
    mad = np.median(np.abs(original_values - median))

    # Compute bounds for capping
    lower = median - k * mad
    upper = median + k * mad

    # Apply capping logic (but don't mutate yet)
    capped_values = original_values.clip(lower, upper)

    # Count modified rows using float-safe comparison
    capped_rows = (np.abs(original_values - capped_values) > 1e-8).sum()

    # Print detailed report
    print(f"Outlier Capping Summary for '{column}':")
    print(f"  Median: {median:.2f}")
    print(f"  MAD: {mad:.2f}")
    print(f"  Boundaries: [{lower:.2f}, {upper:.2f}]")
    print(f"  Rows capped: {capped_rows} ({(capped_rows / len(df)) * 100:.2f}%)")

    # Update column with capped values
    df[column] = capped_values