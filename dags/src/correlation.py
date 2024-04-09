import os
from prompt_toolkit.shortcuts import yes_no_dialog
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sns.set_style('whitegrid')

# Define the project directory path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the input and output file paths
INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'X_train_final.parquet')
IMAGE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', "images", 'correlation_heatmap.png')
CORR_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', "correlation_matrix.parquet")

# Define the correlation threshold
THRESHOLD_CORR = 0.8

# Define the output file paths as a tuple
OUTPUT_PICKLE_PATH = (IMAGE_PATH, CORR_PICKLE_PATH)

# Function to calculate correlation
def correlation(in_path=INPUT_PICKLE_PATH, out_path=OUTPUT_PICKLE_PATH, correlation_threshold=THRESHOLD_CORR):

    """
    
    Function: correlation
    
    Description:
    This function calculates the correlation matrix for the input DataFrame, creates a heatmap visualization of the correlation matrix, 
    and saves both the heatmap image and the correlation matrix as a Parquet file. It also validates the input and output paths and the 
    correlation threshold. If the input file path is invalid or the DataFrame cannot be loaded, it raises FileNotFoundError or TypeError, 
    respectively. It also raises TypeError if the output paths are not strings, and ValueError if the correlation threshold is not between 0 and 1.
    
    Parameters:
    - in_path (str): Path to the input Parquet file containing the DataFrame. Default is INPUT_PICKLE_PATH.
    - out_path (tuple of str): Tuple containing paths to save the heatmap image and the correlation matrix Parquet file, respectively. Default is OUTPUT_PICKLE_PATH.
    - correlation_threshold (float): Threshold for correlation. Default is THRESHOLD_CORR.
    
    Returns:
    None
    """
    data = None
    try:
        # Read data from the input file
        data = pd.read_parquet(in_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Unable to locate the file at {in_path}.") from None

    if not isinstance(data, pd.DataFrame):
        raise TypeError("The file could not be loaded as DataFrame.") from None

    print(data)

    try:
        assert isinstance(out_path[0], str)
    except AssertionError as ae:
        raise TypeError("The path for saving image should be a string!") from ae

    try:
        assert isinstance(out_path[0], str)
    except AssertionError as ae:
        raise TypeError("The path for saving Parquet file should be a string!") from ae

    if not 0 < correlation_threshold < 1:
        raise ValueError("The correlation threshold should be between 0 and 1.")

    corr = data.corr()

    # Create a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True

    # Define custom colormap
    colors = ['#ff6200', '#ffcaa8', 'white', '#ffcaa8', '#ff6200']
    my_cmap = LinearSegmentedColormap.from_list('custom_map', colors, N=256)

    # Create correlation heatmap
    fig = plt.figure(figsize=(10, 10))
    plt.title(f'Correlation Matrix, Threshold:{correlation_threshold}', fontsize=14)
    sns.heatmap(corr, mask=mask, cmap=my_cmap, annot=True, center=0, fmt='.2f', linewidths=2)

    # Save the heatmap image
    save_heatmap(fig, out_path[0])

    # Save correlations as a Parquet file
    save_correlations_as_parquet(corr, out_path[1])

# Function to save heatmap image
def save_heatmap(fig, path):

    """
    Function: save_heatmap
    
    Description:
    This function saves the provided heatmap figure as an image file at the specified path. 
    It first checks if the directory exists and creates it if it doesn't. If the file already exists, it prompts 
    the user to confirm overwriting. If the user confirms, it saves the file; otherwise, it prints a message 
    indicating the failure to save.
    
    Parameters:
    - fig (matplotlib.figure.Figure): The figure object representing the heatmap.
    - path (str): The path to save the heatmap image.
    
    Returns:
    None
    
    """
   
    try:
        p = os.path.dirname(path)
        if not os.path.exists(p):
            os.makedirs(p)
        fig.savefig(path)
        print(f"The file has been successfully saved at the path: {path}.")
    except FileExistsError as fe:
        result = yes_no_dialog(
            title='Error: File Already Exists',
            text=f"A file with the same name already exists. Please close it and try again. Error: {fe}.").run()
        if result:
            fig.savefig(path)
        else:
            print(f"Unable to save the file at the path: {path}.")

# Function to save correlations as Parquet file
def save_correlations_as_parquet(data, path):

    """
    Function: save_correlations_as_parquet
    
    Description:
    This function saves the provided DataFrame as a Parquet file at the specified path. 
    It first checks if the directory exists and creates it if it doesn't. If the file already exists, 
    it prompts the user to confirm overwriting. If the user confirms, it saves the file; otherwise, 
    it prints a message indicating the failure to save. It also checks if the input data is a DataFrame before saving.
    
    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the correlation matrix.
    - path (str): The path to save the Parquet file.
    
    Returns:
    None
    """
   
    try:
        p = os.path.dirname(path)
        if not os.path.exists(p):
            os.makedirs(p)
        data.to_parquet(path)
        print(f"The file has been successfully saved at the path: {path}.")
    except AttributeError as ae:
        raise AttributeError("Unable to execute 'to_parquet' method as the object is not a DataFrame.") from ae
    except FileExistsError as fe:
        result = yes_no_dialog(
            title='Error: File Already Exists',
            text=f"A file with the same name already exists. Please close it and try again. Error: {fe}.").run()
        if result:
            data.to_parquet(path)
        else:
            print(f"Unable to save the file at the path: {path}.")
