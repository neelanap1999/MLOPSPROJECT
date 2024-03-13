import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from src.notification import notify_failure

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_outlier.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_income_normalization.pkl')

def normalize_amount(input_pickle_path=INPUT_PICKLE_PATH,
                            output_pickle_path=OUTPUT_PICKLE_PATH):


    if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"No data found at the specified path: {input_pickle_path}")
        notify_failure(f"No data found at the specified path: {input_pickle_path}")

    df['loan_amnt'] = np.log(df['loan_amnt'])
    df['annual_inc'] = np.log(df['annual_inc']) 

    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved to {output_pickle_path}.")
    return output_pickle_path