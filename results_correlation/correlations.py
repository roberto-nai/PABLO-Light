### IMPORTS ###
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, pearsonr, spearmanr
import re
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent)) # to get the config in parent directory

### LOCAL IMPORTS ###
from config import config_reader

### GLOBALS ###

yaml_config = config_reader.config_read_yaml("config.yml", "config")
datasets_dir = str(yaml_config["DATASETS_POSITION_DIR"])    # Dataset directory with custom encoded event log
correlation_dir = str(yaml_config["CORRELATIONS_DIR"])      # Dataset directory with correlation results

search_name = "synthetic_log_001"  # <-- INPUT: name of the dataset

plot = False # <-- INPUT: if plots are needed

# Define columns to ignore
ignored_columns = ["case:concept:name", "Query_CaseID", "trace_id", "likelihood"]  # <-- INPUT
        
# Define target column
target_column = "case:label" # <-- INPUT

def find_matching_csv_files(directory: Path, name: str) -> list:
    """
    Finds all CSV files in the given directory that contain the specified name
    and the substring "_join_".

    Parameters:
        directory (Path): The directory to search in.
        name (str): The base name to look for.

    Returns:
        list: A list of matching file paths as Path objects.
    """
    return [file for file in directory.glob("*.csv") if name in file.name and "_join_" in file.name]

def extract_activity_name(file_name: str) -> str:
    """
    Extracts the activity name from the file name.
    The activity name is the value after 'A-' and before '_R_'.

    Parameters:
        file_name (str): The filename to parse.

    Returns:
        str: Extracted activity name, or "UNKNOWN" if not found.
    """
    match = re.search(r"A-([A-Za-z0-9-_]+)_R_", file_name)
    return match.group(1) if match else "UNKNOWN"

def analyse_correlations(df: pd.DataFrame, target_column: str, ignored_columns: list, file_name: str, plot: bool) -> pd.DataFrame:
    """
    Analyses the correlation between numerical features and the target column.
    Performs Pearson correlation if normality is met, otherwise uses Spearman correlation.

    Parameters:
        df (pd.DataFrame): The dataset.
        target_column (str): The column to compare against.
        ignored_columns (list): Columns to exclude from analysis.
        file_name (str): The name of the processed file.
        plot (bool): If the plots are needed.

    Returns:
        pd.DataFrame: A DataFrame with correlation results.
    """
    results = []  # Store correlation results
    activity_name = extract_activity_name(file_name)  # Extract activity name from filename

    # Remove ignored columns
    df_filtered = df.drop(columns=ignored_columns, errors="ignore")
    
    # Ensure numerical values
    df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')

    # Iterate over each feature
    for col in df_filtered.columns:
        if col != target_column:
            valid_data = df_filtered[[col, target_column]].dropna()
            
            if len(valid_data) > 1:
                # Check if the column has variance (std > 0)
                std = valid_data[col].std()
                if valid_data[col].std() == 0:
                    print(f"Skipping '{col}': No variance (constant column).")
                
                if plot == True:
                    # Scatter plot to check linearity
                    plt.figure(figsize=(6, 4))
                    sns.scatterplot(x=valid_data[col], y=valid_data[target_column])
                    plt.title(f"Scatter plot of {col} vs {target_column}")
                    plt.xlabel(col)
                    plt.ylabel(target_column)
                    # plt.show()

                    # Histogram for distribution check
                    plt.figure(figsize=(6, 4))
                    sns.histplot(valid_data[col], kde=True, bins=30)
                    plt.title(f"Histogram of {col}")
                    # plt.show()
                    
                # Shapiro-Wilk test for normality
                stat, p_value = shapiro(valid_data[col])
                print(f"Shapiro-Wilk test for {col}: p-value = {p_value:.4f}")
                
                correlation_value = 0
                p_corr = 0

                if p_value >= 0.05:
                    # Pearson correlation if normal
                    print(f"Pearson applicable for column '{col}'.")
                    # correlation, p_corr = pearsonr(valid_data[col], valid_data[target_column])
                    correlation_value, _ = pearsonr(valid_data[col], valid_data[target_column])
                    correlation_type = "Pearson"
                else:
                    print(f"Pearson not applicable for column '{col}', using Spearman.")
                    correlation_value, _ = spearmanr(valid_data[col], valid_data[target_column])
                    correlation_type = "Spearman"

                # Store results
                results.append({
                    "file_name": file_name,
                    "activity_name": activity_name,
                    "std": round(std,3),
                    "correlation_type": correlation_type,
                    "target_column": target_column,
                    "feature": col,
                    "correlation_value": correlation_value
                })

    # Return DataFrame with results
    return pd.DataFrame(results)

def main():
    """Main function to find matching files and perform correlation analysis."""
    
    directory = Path(datasets_dir) 

    # Find matching CSV files
    matching_files = find_matching_csv_files(directory, search_name)
    
    if not matching_files:
        print(f"No matching files found for '{search_name}'.")
        return
    else: 
        print(f"Matching files found for '{search_name}': {len(matching_files)}.")
    
    all_results = []  # Store all correlation results

    # Process each matching file
    for file_path in matching_files:
        print(f"Processing file: {file_path}")
        
        # Load dataset
        df = pd.read_csv(file_path)

        # Perform correlation analysis
        correlation_results = analyse_correlations(df, target_column, ignored_columns, file_path.name, plot)
        all_results.append(correlation_results)
    print()

    # Combine results from all files
    if all_results:
        print("Comining all results")
        final_results_df = pd.concat(all_results, ignore_index=True)
        final_results_df = final_results_df.sort_values(by=["activity_name", "feature"])
        print(final_results_df.head())
        file_out = f"{search_name}_correlations.csv"
        path_out = Path(correlation_dir) / file_out
        print("Saving correlation results to:", path_out)
        final_results_df.to_csv(path_out, sep=";", index=False)
    else:
        print("No results for", directory)
    print()

# Run main function
if __name__ == "__main__":
    main()