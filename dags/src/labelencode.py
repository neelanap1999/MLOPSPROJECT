import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'after_income_normalization.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'encoder_output.parquet')

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

def encode(in_path=INPUT_PICKLE_PATH, out_path=OUTPUT_PICKLE_PATH):

    df = pd.read_pickle(in_path)

    logger.info(">>>>>>>>>>>>>>>>>>>>>>>> Started Encode Task <<<<<<<<<<<<<<<<<<<<<<<<<<")

    dic={' 36 months':36, ' 60 months':60}
    df['term']=df.term.map(dic)

    # Applying mapping function to the column
    df['emp_length'] = df['emp_length'].map(map_years)

    # Label encoding for 'loan_status' column
    le=LabelEncoder()
    df['loan_status']=le.fit_transform(df['loan_status'])
    print(f"Data saved to df after encoding.")
    
    df.to_parquet(out_path)

    logger.info(">>>>>>>>>>>>>>>>>>>>>>>> Encode Task Completed <<<<<<<<<<<<<<<<<<<<<<<<<<")

    return out_path


