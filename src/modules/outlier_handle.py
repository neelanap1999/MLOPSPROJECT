import os
import pickle
from sklearn.ensemble import IsolationForest
import logging

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(LOG_PATH)

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_transform_emp_length.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_outlier.pkl')

def handle_outliers(df):
    """
    Identifying and handling outliers in a DataFrame using Isolation Forest algorithm,
    and saving the cleaned DataFrame to a pickle file.

    Args:
        input_pickle_path (str): The file path to the input pickle file containing the DataFrame (Defaults to INPUT_PICKLE_PATH).
        output_pickle_path (str): The file path to save the output pickle file containing the cleaned DataFrame (Defaults to OUTPUT_PICKLE_PATH).

    Returns:
        str: The file path where the cleaned DataFrame is saved.
        
    Raises:
        FileNotFoundError: If no data is found at the specified input path.
    """
    logger.info(f">>>>>>>>> Started handling outliers <<<<<<<<<<<<<<<<.")

    model = IsolationForest(contamination=0.05, random_state=0)

    df['Outlier_Scores'] = model.fit_predict(df.drop(['loan_status','issue_d'],axis=1).to_numpy())
    df['Is_Outlier'] = [1 if x == -1 else 0 for x in df['Outlier_Scores']]
    df_cleaned = df[df['Is_Outlier'] == 0]
    df_cleaned.drop(['Is_Outlier','Outlier_Scores'],axis=1,inplace=True)

    logger.info(f">>>>>>>>>> Data saved as dataframe after handling outliers <<<<<<<<<<")
    return df_cleaned