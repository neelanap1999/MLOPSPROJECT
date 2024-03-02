import pickle
import os
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DEFAULT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data',
                                   'processed', 'initial.pkl')
DEFAULT_EXCEL_PATH = os.path.join(PROJECT_DIR, 'data', 'initial.csv')

def load_data(pickle_path=DEFAULT_PICKLE_PATH, excel_path=DEFAULT_EXCEL_PATH):
    df = None
    # Check if pickle file exists
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            df = pickle.load(file)
        print(f"Data loaded successfully from {pickle_path}.")
    # If pickle doesn't exist, load from Excel
    elif os.path.exists(excel_path):
        df = pd.read_csv(excel_path)
        print(f"Data loaded from {excel_path}.")
    else:
        error_message = f"No data found in the specified paths: {pickle_path} or {excel_path}"
        print(error_message)
        raise FileNotFoundError(error_message)
    # Save the data to pickle for future use (or re-save it if loaded from existing pickle)
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved to {pickle_path} for future use.")
    return pickle_path


