"""
dataframe_operations.py

Description: Utilities on pandas dataframe structures.

Changelog:
[2024-11-25]: Added get_distinct_column_values.
[2024-02-07]: Added find_activity_position_by_index.
[2024-02-07]: Added get_case_lengths (useful for prefix_compute = 0).
"""

import pandas as pd

def columns_unique_values(df_in: pd.DataFrame, prefix_col_name: str) -> list:
    """
    Returns a list of unique string values from specific columns in a DataFrame. 
    The columns are selected based on their names starting with a given prefix.

    Parameters:
        df_in (pd.DataFrame): The DataFrame to analyse.
        prefix_col_name (str): The prefix to filter column names.

    Returns:
        list: A list of unique string values with duplicates removed.
    """
    # Get the list of column names matching the prefix
    matching_columns = [col for col in df_in.columns if col.startswith(prefix_col_name)]
    matching_columns_len = len(matching_columns)
    # print(f"Columns matching prefix '{prefix_col_name}': {matching_columns}")
    print(f"Columns matching prefix '{prefix_col_name}': {matching_columns_len}")

    # If no matching columns, return an empty list
    if not matching_columns:
        return []

    # Select the data from matching columns
    selected_columns = df_in[matching_columns].values

    # Flatten the selected columns to extract all values and ensure all values are strings -> map(str,...)
    unique_values = set(map(str, selected_columns.flatten()))

    # Remove duplicates by converting to a set and back to a list, then sort
    result = sorted(list(unique_values))

    return result

def encode_activities_positions_and_repetitions_v1(
    df_in: pd.DataFrame, 
    activity_query: str, 
    case_id_col: str = "case:concept:name", 
    activity_col: str = "concept:name", 
    activity_null: int = 0, 
    df_attribute_columns: list = None
) -> pd.DataFrame:
    """
    Process the dataframe to compute activity_first, activity_last, and activity_repetition for a given activity query. Optionally includes additional columns from the first occurrence of each case_id.
    v1: per normal dataframe.

    Parameters:
    - df_in: pd.DataFrame - The input dataframe.
    - activity_query: str - The activity to query in the dataframe.
    - case_id_col: str, default "case:concept:name" - The column name representing case IDs.
    - activity_col: str, default "concept:name" - The column name representing activity names.
    - activity_null: int or float, default 0 - The value to use if the activity is not found in a case.
    - df_attribute_columns: List[str], optional - Additional columns to include in the result, taken from the first occurrence of each case_id.

    Returns:
    - pd.DataFrame - A dataframe with columns [case_id_col, activity_first, activity_last, activity_repetition, *df_attribute_columns].
    """
    # Create a dictionary to store results
    results = {
        case_id_col: [],
        "activity_first": [],
        "activity_last": [],
        "activity_repetition": []
    }
    
    # Add df_attribute_columns to the results dictionary
    if df_attribute_columns:
        for col in df_attribute_columns:
            results[col] = []

    # Group by the case ID column for better performance
    grouped = df_in.groupby(case_id_col)
    
    for case_id, group in grouped:
        # Reset index to ensure positions are relative to the start of the case
        group = group.reset_index(drop=True)
        
        # Find the positions where the activity matches the query
        activity_positions = group.index[group[activity_col] == activity_query].tolist()
        
        if activity_positions:
            # If the activity is found, calculate first, last, and repetition count
            activity_first = activity_positions[0] + 1  # 1-indexed
            activity_last = activity_positions[-1] + 1  # 1-indexed
            activity_repetition = len(activity_positions)
        else:
            # If the activity is not found, use activity_null
            activity_first = activity_null
            activity_last = activity_null
            activity_repetition = activity_null

        # Append the results
        results[case_id_col].append(case_id)
        results["activity_first"].append(activity_first)
        results["activity_last"].append(activity_last)
        results["activity_repetition"].append(activity_repetition)
        
        # Append values for df_attribute_columns, taking the first occurrence for each case_id
        if df_attribute_columns:
            for col in df_attribute_columns:
                if col in group.columns:
                    # If the column exists, append its value from the first occurrence
                    results[col].append(group.iloc[0][col])
                else:
                    # If the column does not exist, append a placeholder (e.g., None or NaN)
                    results[col].append(None)

    # Convert the results into a dataframe
    return pd.DataFrame(results)

def encode_activities_positions_and_repetitions_v2(
    df_in: pd.DataFrame,
    activity_query: str,
    case_id_col: str = "trace_id",
    activity_col_prefix: str = "prefix_",
    activity_null: int = 0
) -> pd.DataFrame:
    """
    Process the dataframe to compute activity_first, activity_last, and activity_repetition for a given activity query. Resets the counter for each unique case_id.
    v2: for prefix dataframe.

    Parameters:
    - df_in: pd.DataFrame - The input dataframe.
    - activity_query: str - The activity to query in the dataframe.
    - case_id_col: str, default "trace_id" - The column name representing case IDs.
    - activity_col_prefix: str, default "prefix_" - The prefix for activity columns.
    - activity_null: int, default 0 - The value to use if the activity is not found in a case.

    Returns:
    - pd.DataFrame - A dataframe with columns [case_id_col, activity_first, activity_last, activity_repetition].
    """
    results = {
        case_id_col: [],
        "activity_first": [],
        "activity_last": [],
        "activity_repetition": []
    }

    # Filter activity columns based on the prefix
    activity_columns = [col for col in df_in.columns if col.startswith(activity_col_prefix)]

    # Group by the case ID column for better performance
    grouped = df_in.groupby(case_id_col)

    for case_id, group in grouped:
        activity_positions = []
        for idx, col in enumerate(activity_columns):
            if group[col].eq(activity_query).any():
                activity_positions.append(idx + 1)  # Convert to 1-based indexing

        if activity_positions:
            activity_first = activity_positions[0]
            activity_last = activity_positions[-1]
            activity_repetition = len(activity_positions)
        else:
            activity_first = activity_null
            activity_last = activity_null
            activity_repetition = activity_null

        # Append results
        results[case_id_col].append(case_id)
        results["activity_first"].append(activity_first)
        results["activity_last"].append(activity_last)
        results["activity_repetition"].append(activity_repetition)

    # Convert results dictionary into a dataframe
    return pd.DataFrame(results)

def dictionary_unique_values(data: dict, exclude_keys: list) -> list:
    """
    Extracts unique values from a dictionary while excluding specific keys.

    Parameters:
    data (dict): A dictionary where each key maps to another dictionary 
                 with integer keys and string values.
    exclude_keys (list): A list of keys to exclude from the extraction process.

    Returns:
    list: A sorted list of unique string values from the dictionary, excluding specified keys.
    """
    # Set to store unique values
    distinct_values = set()

    # Iterate over the dictionary
    for key, mapping in data.items():
        if key not in exclude_keys:  # Skip keys that are in the exclusion list
            distinct_values.update(mapping.values())

    # Return a sorted list of unique values
    return sorted(distinct_values)

def get_distinct_column_values(df: pd.DataFrame, col_name: str, n: int = 0) -> list:
    """
    Returns the first n distinct values from the specified column of a dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe to extract values from.
        col_name (str): The name of the column to extract values from.
        n (int): The number of distinct values to retrieve.

    Returns:
        List: A list containing the first n distinct values of the specified column.

    Raises:
        ValueError: If the specified column does not exist in the dataframe.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the dataframe.")
    
    if n == 0:
        return df[col_name].drop_duplicates().tolist()
    else:
        return df[col_name].drop_duplicates().head(n).tolist()
    
def find_activity_position_by_index(df: pd.DataFrame, trace_id: int, activity: str, prefix_column: str = "prefix_") -> int:
    """
    Finds the position of a given activity within the prefix columns for a specified trace_id.

    Parameters:
        df (pd.DataFrame): The dataframe containing the trace data.
        trace_id (int): The identifier of the trace to search within.
        activity (str): The activity to locate in the prefix columns.
        prefix_column (str, optional): The prefix used to identify relevant columns. Defaults to "prefix_".

    Returns:
        int: The position of the activity within the prefix columns, or -1 if not found.
    """
    prefix_columns = [col for col in df.columns if col.startswith(prefix_column)]

    row = df[df["trace_id"] == trace_id]
    if row.empty:
        return -1  # Trace ID not found

    for i, col in enumerate(prefix_columns, start=1):
        if row[col].values[0] == activity:
            return i  # Position is 1-based

    return -1  # Activity not found

def find_activity_position_by_name(df: pd.DataFrame, activity_column: str, activity: str) -> int:
    """
    Finds the position of a given activity within the specified activity column.

    Parameters:
        df (pd.DataFrame): The dataframe containing the event log data.
        activity_column (str): The column name that contains the activities.
        activity (str): The activity to locate in the column.

    Returns:
        int: The position of the activity in the sequence (1-based), or -1 if not found.
    """
    positions = df.index[df[activity_column] == activity].tolist()
    return positions[0] + 1 if positions else -1

def get_case_lengths(event_log: pd.DataFrame, group_column: str) -> dict:
    """
    Returns a dictionary with unique values from the specified column as keys and their sequence length as values.
    
    Parameters:
        event_log (pd.DataFrame): DataFrame containing the event log.
        group_column (str): Column name to group by (e.g., 'case:concept:name').
    
    Returns:
        dict: A dictionary where keys are unique values from the specified column and values are the count of events.
    """
    case_counts = event_log[group_column].value_counts().to_dict()
    return case_counts