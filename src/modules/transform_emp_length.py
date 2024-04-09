import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Configure logging
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_dummies.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_transform_emp_length.pkl')

# Define mapping function
def map_years(years):
    """
    Mapping employment length values to numerical values.

    Args:
        years (str or int): The employment length value.

    Returns:
        int or np.nan: The mapped numerical value for employment length, or np.nan if input is NaN.

    Raises:
        None
    """
    if pd.isna(years):  # Handle NaN values
        return np.nan
    elif isinstance(years, int):  # If already an integer, return as is
        return years
    elif years == '< 1 year':
        return 0
    elif years == '10+ years':
        return 10
    else:
        return int(years.split()[0])

def emp_len_transform(input_pickle_path=INPUT_PICKLE_PATH,
                            output_pickle_path=OUTPUT_PICKLE_PATH):
    """
    Transforming the 'emp_length' column in a DataFrame by mapping employment length values to numerical values,
    and saving the updated DataFrame to a pickle file.

    Args:
        input_pickle_path (str): The file path to the input pickle file containing the DataFrame (Defaults to INPUT_PICKLE_PATH).
        output_pickle_path (str): The file path to save the output pickle file containing the updated DataFrame (Defaults to OUTPUT_PICKLE_PATH).

    Returns:
        str: The file path where the updated DataFrame is saved.
        
    Raises:
        FileNotFoundError: If no data is found at the specified input path.
    """
    if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"No data found at the specified path: {input_pickle_path}")

    # Applying mapping function to the column
    df['emp_length'] = df['emp_length'].map(map_years)
    df['emp_length'] = df['emp_length']

    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved to {output_pickle_path}.")
    return output_pickle_path