"""
csv_to_xes.py

Standalone module for converting event logs from CSV to XES

"""
### IMPORTS ###

import pandas as pd
from pathlib import Path
import pm4py
# pip install -U pm4py

### GLOBALS ###

input_csv = "synthetic_log_001.csv"  # <-- INPUT: file to be converted
csv_sep = ","  # <-- INPUT: CSV file separator

# Mandatory values
caseid_col = "case:concept:name" 
activity_col = "concept:name" 
timestamp_col = "time:timestamp"

# Optional values
label_col = "label" # <-- INPUT: the label is set as trace attribute (set it as None or "" to skip it)

# Optional values
trace_cols = [] #  # <-- INPUT: columns to be moved as trace attributes (empty list to skip it)

### FUNCTIONS ###

def csv_to_xes(input_csv_path: str, output_xes_path: str, caseid_col: str, activity_col: str, timestamp_col: str, label_col: str, trace_cols: list) -> None:
    """
    Reads a CSV file with columns and converts the contents into an XES file, then saves the file to the specified path.

    Parameters
        - input_csv_path: Path to the input CSV file.
        - output_xes_path: Path for the resulting XES file.
        - caseid_col: Name of the column identifying the case ID.
        - activity_col: Name of the column identifying the activity.
        - timestamp_col: Name of the column identifying the timestamp.
        - label_col: Name of the column identifying the label (optional).
        - trace_cols: list of the columns to be moved as trace attribute (optional).

    Returns
        - None
    """

    print("Reading CSV data")
    # 1. Read the CSV into a Pandas DataFrame
    df = pd.read_csv(input_csv_path, sep=",")

    # 2. Convert the 'time:timestamp' column to datetime format
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 2.1 Convert the other main columns in string format
    df[caseid_col] = df[caseid_col].astype(str)
    df[activity_col] = df[activity_col].astype(str)

    # 2.2 Ensure 'label' (optional) is a string
    if label_col in df.columns:
        df[label_col] = df[label_col].astype(str)
    print()

    print("Creating XES file")
    # 3. Convert the DataFrame into a pm4py event log.
    # Specify the key columns for case_id, activity, and timestamp.
    # All other columns will be included as additional event attributes (by pm4py default).
    log = pm4py.convert_to_event_log(
        df,
        case_id_key=caseid_col,
        activity_key=activity_col,
        timestamp_key=timestamp_col
    )
    print()

    # 4. Move label_col (if present) from the event level to the trace level.
    print("Moving label column to trace level")
    if label_col is not None or label_col != "":
        print(f"Label column available with name '{label_col}'")
        for trace in log:
            if len(trace) > 0 and label_col in trace[0]:
                label_val = trace[0][label_col]
                trace.attributes[label_col] = label_val
                # Remove label_col from each event, if present
                for event in trace:
                    if label_col in event:
                        del event[label_col]
    print()

    # 5. Move trace_cols (if present) from the event level to the trace level.
    print("Moving columns to trace level")
    trace_cols_len = len(trace_cols)
    print("Columns to be moved:", trace_cols_len)
    if trace_cols_len > 0:
        for col_name in trace_cols:
            print("Moving column:", col_name)
            for trace in log:
                if len(trace) > 0 and col_name in trace[0]:
                    label_val = trace[0][col_name]
                    trace.attributes[col_name] = label_val
                    # Remove col_name from each event, if present
                    for event in trace:
                        if col_name in event:
                            del event[col_name]
    print()

    # 6. Export the event log in XES format
    print("Saving the XES file")
    pm4py.write_xes(log, output_xes_path)
    print()

### MAIN ###

if __name__ == "__main__":
    print()
    print("*** CSV to XES conversion ***")
    output_xes = f"{Path(input_csv).stem}.xes"
    print(f"Converting input CSV '{input_csv}'")
    print()
    csv_to_xes(input_csv, output_xes, caseid_col, activity_col, timestamp_col, label_col, trace_cols)
    print(f"Conversion completed: XES file saved as '{output_xes}'")
    print()