import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_income_normalization.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_transform_emp_length.pkl')

# Define mapping function
def map_years(years):
    if pd.isna(years):  # Handle NaN values
        return np.nan
    elif years == '< 1 year':
        return 0
    elif years == '10+ years':
        return 10
    else:
        return int(years.split()[0])

def emp_len_transform(input_pickle_path=INPUT_PICKLE_PATH,
                            output_pickle_path=OUTPUT_PICKLE_PATH):

    if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"No data found at the specified path: {input_pickle_path}")

    # Applying mapping function to the column
    df['emp_length'] = df['emp_length'].map(map_years)
    df['emp_length'] = df['emp_length'].astype(int)

    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved to {output_pickle_path}.")
    return output_pickle_path