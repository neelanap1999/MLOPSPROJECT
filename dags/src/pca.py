import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from prompt_toolkit.shortcuts import yes_no_dialog

# Loading Config File
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global variables
INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR,'data', 'processed', 'scaler_output.parquet')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR,'data', 'processed', 'pca_output.parquet')
NOT_COLUMNS = None
THRESHOLD_CVR = 0.8

def pc_analyzer(in_path=INPUT_PICKLE_PATH, out_path=OUTPUT_PICKLE_PATH,
                drop_cols=NOT_COLUMNS, cvr_thresh=THRESHOLD_CVR):

    data = None

    # File Loading
    try:
        data = pd.read_parquet(in_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Unable to locate the file at {in_path}.") from None

    # Check datatype
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Unexpected data type. Unable to load DataFrame correctly.") from None

    # Check if all required columns exist in DataFrame
    try:
        assert all(col in data.columns for col in drop_cols)
    except AssertionError as exc:
        raise KeyError("Required column not found in the DataFrame.") from exc

    # Check if the number of columns is less than 4, in which case PCA is not required
    if len(data.columns) < 4:
        raise ValueError("Number of columns is less than 4. PCA analysis is unnecessary.")

    # Value check for CVR_THRESHOLD
    if not 0 < cvr_thresh < 1:
        raise ValueError("Invalid value for CVR_THRESHOLD. It should be between 0 and 1.")

    # Selecting columns in data for PCA
    data.set_index('CustomerID', inplace=True)
    data = data.drop(drop_cols, axis=1)

    print(data)

    n = pca_(data, cvr_thresh)
    pca = PCA(n_components=n).fit(data)

    # Getting post-PCA data
    reduced_data = pca.transform(data)
    columns = [f'PC{i+1}' for i in range(pca.n_components_)]
    pca_transformed_data = pd.DataFrame(reduced_data, columns=columns)
    pca_transformed_data.index = data.index
    print(pca_transformed_data)

    # Saving data as parquet
    try:
        p = os.path.dirname(out_path)
        if not os.path.exists(p):
            os.makedirs(p)
        pca_transformed_data.to_parquet(out_path)
        print(f"File saved successfully at the specified path: {out_path}.")
    except FileExistsError:
        result = yes_no_dialog(
            title='File Already Exists',
            text="The file you are attempting to save already exists. Please close it to overwrite.").run()
        if result:
            pca_transformed_data.to_parquet(out_path)
            print(f"File saved successfully at the specified path: {out_path}.")
        else:
            print(f"Unable to save the file at the specified path: {out_path}.")

    return out_path

def pca_(data, thresh):
    n_components_range = np.arange(2, 9)
    n = []
    cumulative_var_ratio = [0]
    for n_components in n_components_range:
        if cumulative_var_ratio[-1] <= thresh:
            pca_check = PCA(n_components=n_components).fit(data)
            var_ratio = pca_check.explained_variance_ratio_
            cumulative_var_ratio.append(np.cumsum(var_ratio)[n_components-1])
            n.append(n_components)
            print(n[-1], cumulative_var_ratio[-1])
        elif n_components == n_components_range[-1]:
            n = n_components
            break
        else:
            n = n_components
            break
    print(n, cumulative_var_ratio)
    return n
