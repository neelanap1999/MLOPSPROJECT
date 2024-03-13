import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from prompt_toolkit.shortcuts import yes_no_dialog

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'after_outlier_treatment.pickle')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'scaler_output.parquet')
S_COLS = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
          'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']

def scaler(in_path=INPUT_PICKLE_PATH, out_path=OUTPUT_PICKLE_PATH, cols_to_scale=S_COLS):
    
    data = None

    try:
        data = pd.read_pickle(in_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {in_path}.") from None

    try:
        assert isinstance(out_path, str)
    except AssertionError as ae:
        raise TypeError("Save Path should be a String!") from ae

    if not isinstance(data, pd.DataFrame):
        raise TypeError("File did not load DataFrame correctly.") from None

    try:
        assert all(col in data.columns for col in cols_to_scale)
    except AssertionError as exc:
        raise KeyError("Column listed in to be Standardized \
        Columns not found in Dataframe.") from exc

    cols_to_keep = [col for col in data.columns if col not in cols_to_scale]

    scaler_std = StandardScaler()
    data_std = scaler_std.fit_transform(data[cols_to_scale])
    std_df = pd.DataFrame(data_std, columns=cols_to_scale)

    scaled_data = pd.concat([std_df, data[cols_to_keep]], axis=1)

    try:
        p = os.path.dirname(out_path)
        if not os.path.exists(p):
            os.makedirs(p)
        scaled_data.to_parquet(out_path)
        print(f"File saved successfully at Path: {out_path}.")
    except FileExistsError:
        result = yes_no_dialog(
            title='File Exists Error',
            text="Existing file in use. Please close to overwrite the file. Error:.").run()
        if result:
            scaled_data.to_parquet(out_path)
            print(f"File saved successfully at Path: {out_path}.")
        else:
            print(f"Could not save File at Path: {out_path}.")
    return out_path