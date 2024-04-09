"""
Function to download and ingest the data file
"""
import os
import gdown
import logging
from src.utils.config import load_config
# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(LOG_PATH)

DEFAULT_FILE_URL = "https://drive.google.com/file/d/1NAn7I7iJGxy2AhrmfkVdo37GY1dtGLzw/view?usp=sharing"
config = load_config()

def ingest_data(file_url=DEFAULT_FILE_URL):
    """
    Function to download file from URL
    Args:
        file_url: URL of the file, A default is used if not specified
    Returns:
        csvfile_path: The zipped file path to the data
    """
     # Set the root directory variable using a relative path
    root_dir = config.data_ingestion.root_dir

    # Path to store the csv
    csvfile_path=os.path.join(root_dir,'initial_data.csv')
    file_id = file_url.split("/")[-2]
    prefix = 'https://drive.google.com/uc?/export=download&id='
    logger.info("Download data...")

    # Check if the request was successful (status code 200)
    try:
        # Save file to data
        gdown.download(prefix+file_id,csvfile_path)
        # print(f"File downloaded successfully. Zip file available under {csvfile_path}")
    except Exception as e:
            raise e
    return csvfile_path