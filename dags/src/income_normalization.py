import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_outlier.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_income_normalization.pkl')

def handle_outliers(input_pickle_path=INPUT_PICKLE_PATH,
                            output_pickle_path=OUTPUT_PICKLE_PATH):


    if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"No data found at the specified path: {input_pickle_path}")

    df['loan_amount'] = np.log(df['loan_amount'])
    df['annual_income'] = np.log(df['annual_income']) 

    with open(output_pickle_path, "wb") as file:
        pickle.dump(df_cleaned, file)
    print(f"Data saved to {output_pickle_path}.")
    return output_pickle_path