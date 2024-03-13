import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from prompt_toolkit.shortcuts import yes_no_dialog

# Loading Config File
PAR_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global variables
__INGESTPATH__ = os.path.join(PAR_DIRECTORY, 'data', 'processed', 'after_outlier_treatment.pickle')
__OUTPUTPATH__ = os.path.join(PAR_DIRECTORY, 'data', 'processed', 'scaler_output.parquet')
S_COLS = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
          'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']
N_COLS = ['emp_length','term', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose', 
          'earliest_cr_line', 'initial_list_status', 'application_type', 'address']
COLS = (S_COLS, N_COLS)

def scaler(in_path=__INGESTPATH__, out_path=__OUTPUTPATH__, cols=COLS):
    """
    Global variables(can only be changed through Config file)
    cvr_threshold[float]: cumulative explained variance threshold for variance
    cols(standardize_columns, normalize_columns): tuple of columns to be scaled. 
    """
    # Placeholder for data
    data = None

    # File Loading
    try:
        data = pd.read_pickle(in_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {in_path}.") from None

    try:
        assert isinstance(out_path, str)
    except AssertionError as ae:
        raise TypeError("Save Path should be a String!") from ae

    # Check datatype
    if not isinstance(data, pd.DataFrame):
        raise TypeError("File did not load DataFrame correctly.") from None

    # Check all required columns exist in DF
    try:
        assert all(col in data.columns for col in cols[0])
    except AssertionError as exc:
        raise KeyError("Column listed in to be Standardized \
        Columns not found in Dataframe.") from exc

    try:
        assert all(col in data.columns for col in cols[1])
    except AssertionError as exc2:
        raise KeyError("Column listed in to be Normalized \
            Columns not found in Dataframe.") from exc2

    #print(data)

    # Standardization (-1,1)
    std_scaler = StandardScaler()
    data_std = std_scaler.fit_transform(data[cols[0]])
    df_std = pd.DataFrame(data_std, columns=cols[0])

    # Combining scaled numerical columns with categorical columns
    scaled_data = pd.concat([df_std, data[cols[1]]], axis=1)

    # Saving data as parquet
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

