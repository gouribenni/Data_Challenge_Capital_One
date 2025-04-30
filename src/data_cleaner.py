import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re 

warnings.filterwarnings("ignore")

# Text to number mapping
def text_to_number(text):
    """
    Converts a textual representation of a number into its numeric equivalent.
    
    Example: "Thirty Five" → 35
    """
    mapping = {
        'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5,
        'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9, 'Ten': 10,
        'Twenty': 20, 'Thirty': 30, 'Forty': 40, 'Fifty': 50,
        'Sixty': 60, 'Seventy': 70, 'Eighty': 80, 'Ninety': 90,
        'Hundred': 100, 'Thousand': 1000
    }
    return mapping.get(text.capitalize(), None)

# def clean_numeric_string(value):
#     if pd.isna(value):
#         return np.nan
#     if isinstance(value, str):
#         if value.isdigit():
#             return float(value)
#         value_parts = value.split()
#         num_values = [text_to_number(part) for part in value_parts if text_to_number(part) is not None]
#         return sum(num_values) if num_values else np.nan
#     return value

def clean_numeric_string(value):
    """
    Cleans numeric strings by converting valid digit strings or text-based numbers to float.
    Handles cases like "Thirty Five", "12.5", and ignores invalid formats.
    """
    pattern = r"^-?\d+\.\d+$"
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        if value.isdigit():
            return float(value)
        elif re.match(pattern, value):
            return float(value)
        else:
            value_parts = value.split()
            num_values = [text_to_number(part) for part in value_parts]
            num_values = [num for num in num_values if num is not None]
            if num_values:
                return sum(num_values)
            return np.nan
    return value

def apply_cleaning_to_numeric_columns(df, columns):
    """
    Applies numeric string cleaning to a list of columns in a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - columns (list): List of column names to clean
    
    Returns:
    - pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        df[col] = df[col].apply(clean_numeric_string)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def fill_na_based_on_dtype(df, col, dtype):
    """
    Fills NA values based on the specified data type.
    
    - For 'float': fills with 0
    - For 'string': fills with 'Unknown'
    - For 'date': fills with default timestamp (1970-01-01)
    """
    if dtype == 'float':
        df[col].fillna(0, inplace=True)
    elif dtype == 'string':
        df[col].fillna('Unknown', inplace=True)
    elif dtype == 'date':
        df[col].fillna(pd.Timestamp('1970-01-01'), inplace=True)

def standardize_date(date_str):
    """
    Converts date strings to a standard 'YYYY-MM-DD' format where possible.
    Returns NaT if conversion fails.
    """
    if isinstance(date_str, str):
        if '/' in date_str:
            try:
                return pd.to_datetime(date_str, format='%m/%d/%y').strftime('%Y-%m-%d')
            except:
                return pd.NaT
        return date_str
    return pd.NaT

def analyze_conversion_errors(df, convert_dtypes):
    """
    Tracks and classifies type conversion errors column-wise.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - convert_dtypes (dict): Mapping of column names to target dtypes
    
    Returns:
    - df (pd.DataFrame): DataFrame with additional columns for conversion success
    - error_reports (dict): Summary of error types per column
    - df_errors_list (list): Placeholder for error DataFrames (currently unused)
    """
    def convert_with_error_tracking(original_df, column_name, target_dtype):
        data = original_df.copy()
        if target_dtype in ['float', 'int']:
            converted = pd.to_numeric(data[column_name], errors='coerce')
        elif target_dtype == 'date':
            converted = pd.to_datetime(data[column_name], errors='coerce')
        elif target_dtype == 'string':
            converted = data[column_name].astype(str).replace('nan', pd.NA)
        is_converted = converted.notna()
        original_df[f'{column_name}_conversion_success'] = is_converted
        original_df[f'{column_name}_converted'] = converted
        return original_df

    def classify_conversion_error(original_value, converted_value):
        if pd.isna(converted_value):
            if pd.isna(original_value):
                return ("Original value was NaN", "")
            if isinstance(original_value, str):
                invalid = ''.join([c for c in original_value if not (c.isdigit() or c in ".-")]).strip()
                return ("Invalid characters" if invalid else "Format error", f"Invalid: {invalid or original_value}")
            return ("Other conversion error", str(original_value))
        return ("No error", "")

    def error_analysis_report(data, columns_list):
        reports = {}
        for col in columns_list:
            error_df = data[~data[f"{col}_conversion_success"]].copy()
            error_df['error_details'] = error_df.apply(lambda row: classify_conversion_error(row[col], row[f"{col}_converted"]), axis=1)
            unique_errors = error_df['error_details'].dropna().unique()
            reports[col] = unique_errors
        return reports

    df = df.copy()
    for col, dtype in convert_dtypes.items():
        df = convert_with_error_tracking(df, col, dtype)
    return df, error_analysis_report(df, list(convert_dtypes.keys())), []

def print_error_report(error_reports):
    """
    Prints a structured summary of conversion issues for each column.
    
    Parameters:
    - error_reports (dict): Output from analyze_conversion_errors()
    """

    print("\n====== Conversion Error Summary ======\n")
    table = [
        [col, err[0], (err[1] if len(err[1]) <= 80 else err[1][:77] + "...")]
        for col, errors in error_reports.items() for err in errors if err[0] != "No error"
    ]
    if table:
        print(tabulate(table, headers=["Column", "Error Type", "Details"], tablefmt="fancy_grid"))
    else:
        print("No errors found after conversion.")

def check_nulls(df):
    """
    Identifies columns with null values and calculates their percentages.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    
    Returns:
    - pd.DataFrame: Summary of null counts and percentages
    - list: Columns with at least one null value
    """
    total_rows = len(df)
    null_counts = df.isnull().sum()
    null_summary = pd.DataFrame({
        'Column': df.columns,
        'Total Count': total_rows,
        'Null Count': null_counts.values,
        'Null %': (null_counts.values / total_rows) * 100
    })
    columns_with_nulls = null_summary[null_summary['Null Count'] > 0]
    sorted_nulls = columns_with_nulls.sort_values(by='Null %', ascending=False).reset_index(drop=True)
    return sorted_nulls, columns_with_nulls['Column'].tolist()

def classify_columns(df, cardinality_threshold=20):
    """
    Categorizes DataFrame columns as categorical, ordinal, or numerical.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - cardinality_threshold (int): Threshold below which numeric columns are considered categorical
    
    Returns:
    - dict: Dictionary with keys 'categorical', 'ordinal', and 'numerical'
    """

    col_types = {"categorical": [], "ordinal": [], "numerical": []}
    for col in df.columns:
        dtype = df[col].dtype
        unique_vals = df[col].nunique()
        if np.issubdtype(dtype, np.number):
            if unique_vals <= cardinality_threshold:
                col_types["categorical"].append(col)
            else:
                col_types["numerical"].append(col)
        elif dtype == "object":
            col_types["categorical"].append(col)
        elif dtype == "category":
            col_types["ordinal"].append(col)
        else:
            col_types["categorical"].append(col)
    return col_types

def drop_all_null_columns(df):
    """
    Drops columns that contain only null values.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    
    Returns:
    - pd.DataFrame: DataFrame with null-only columns dropped
    - list: List of dropped column names
    """

    cols_to_drop = df.columns[df.isnull().all()].tolist()
    df_cleaned = df.drop(columns=cols_to_drop)
    print(f"Dropped columns with all NULLs: {cols_to_drop}")
    return df_cleaned, cols_to_drop

def drop_high_null_columns(df, threshold=95):
    """
    Drops columns with a percentage of nulls greater than the given threshold.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - threshold (float): Percent threshold to drop columns
    
    Returns:
    - pd.DataFrame: DataFrame with high-null columns dropped
    - list: List of dropped column names
    """

    total_rows = len(df)
    null_percent = (df.isnull().sum() / total_rows) * 100
    cols_to_drop = null_percent[null_percent > threshold].index.tolist()
    df_cleaned = df.drop(columns=cols_to_drop)
    print(f"Dropped columns with > {threshold}% NULLs: {cols_to_drop}")
    return df_cleaned, cols_to_drop

def check_and_handle_duplicates(df, subset=None, auto_drop=True):
    """
    Identifies and optionally removes duplicate rows from the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - subset (list or None): List of columns to check for duplication
    - auto_drop (bool): Whether to drop duplicates automatically
    
    Returns:
    - pd.DataFrame: Cleaned DataFrame if auto_drop is True
    - dict: Duplicate stats summary
    """

    total_rows = len(df)
    if subset:
        dupes = df[df.duplicated(subset=subset, keep=False)]
        print(f"Checking duplicates based on columns: {subset}")
    else:
        dupes = df[df.duplicated(keep=False)]
        print("Checking fully duplicated rows.")
    dupes_count = dupes.shape[0]
    dupes_pct = (dupes_count / total_rows) * 100
    print(f"Found {dupes_count} duplicates ({dupes_pct:.2f}%).")
    if auto_drop:
        df_cleaned = df.drop_duplicates(subset=subset, keep='first') if subset else df.drop_duplicates(keep='first')
        print(f"Dropped duplicates. New shape: {df_cleaned.shape}")
        return df_cleaned, {"count": dupes_count, "percentage": dupes_pct, "columns_checked": subset or "all columns"}
    return df, {"count": dupes_count, "percentage": dupes_pct, "columns_checked": subset or "all columns"}

def multi_strategy_imputer(df, num_cols=[], cat_cols=[], num_strategy='median', cat_strategy='mode', constant_fill=None):
    """
    Imputes missing values using customizable strategies for numerical and categorical columns.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - num_cols (list): Numerical columns to impute
    - cat_cols (list): Categorical columns to impute
    - num_strategy (str): 'mean' or 'median'
    - cat_strategy (str): 'mode' or 'constant'
    - constant_fill (str): Value to use if cat_strategy is 'constant'
    
    Returns:
    - pd.DataFrame: Imputed DataFrame
    """

    df_copy = df.copy()
    for col in num_cols:
        if df_copy[col].isna().sum():
            fill_val = df_copy[col].median() if num_strategy == 'median' else df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(fill_val)
            print(f"Filled NaNs in {col} with {num_strategy}: {round(fill_val, 2)}")
    for col in cat_cols:
        if df_copy[col].isna().sum():
            if cat_strategy == 'mode':
                fill_val = df_copy[col].mode().iloc[0]
            elif cat_strategy == 'constant':
                fill_val = constant_fill
            df_copy[col] = df_copy[col].fillna(fill_val)
            print(f"Filled NaNs in {col} with {cat_strategy}: {fill_val}")
    return df_copy

def filter_dataset_based_on_user(df, filter_dataset_dict, na_cols):
    """
    Filters a DataFrame based on given user conditions and categorizes columns with NA values.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - filter_dataset_dict (dict): Conditions to apply (column: list of allowed values)
    - na_cols (list): List of columns with nulls
    
    Returns:
    - pd.DataFrame: Filtered DataFrame
    - list: Categorical columns with NA after filtering
    - list: Numerical columns with NA after filtering
    """

    cols_na_after_filtering = []
    numeric_cols_after_filtering = []
    category_cols_after_filtering = []
    for col, values in filter_dataset_dict.items():
        if col not in na_cols:
            df = df[df[col].isin(values)]
        else:
            cols_na_after_filtering.append(col)
    for col in cols_na_after_filtering:
        dtype = df[col].dtype
        if dtype == 'object':
            category_cols_after_filtering.append(col)
        else:
            numeric_cols_after_filtering.append(col)
    return df, category_cols_after_filtering, numeric_cols_after_filtering

def predictive_categorical_imputer(df, cat_col, feature_cols):
    """
    Fills missing values in a categorical column using a RandomForestClassifier.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - cat_col (str): Categorical column with missing values to be imputed
    - feature_cols (list): Feature columns to use for prediction
    
    Returns:
    - pd.Series: Column with imputed values
    """

    df_copy = df.copy()
    df_copy[feature_cols] = df_copy[feature_cols].fillna('Missing')
    feature_encoders = {}
    for col in feature_cols:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        feature_encoders[col] = le
    target_le = LabelEncoder()
    df_copy[cat_col] = df_copy[cat_col].astype(str)
    known = df_copy[~df_copy[cat_col].isin(['nan', 'None', 'NaN'])]
    unknown = df_copy[df_copy[cat_col].isin(['nan', 'None', 'NaN'])]
    X_train = known[feature_cols]
    y_train = target_le.fit_transform(known[cat_col])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    if not unknown.empty:
        X_missing = unknown[feature_cols]
        y_pred = model.predict(X_missing)
        df_copy.loc[unknown.index, cat_col] = target_le.inverse_transform(y_pred)
        print(f"Filled {len(y_pred)} missing values in '{cat_col}' using RandomForest classifier.")
    else:
        print(f"No missing values to impute for '{cat_col}'")
    return df_copy[cat_col]
