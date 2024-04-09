import logging

from src.utils.config import load_config
from src.modules.download_data import ingest_data
from src.modules.dataload import load_data

config = load_config()
Stage_name = "Data ingestion"
SOURCE_URL =  config.data_ingestion.source_url
PICKLE_PATH = config.data_ingestion.raw_pkl

try:
    logging.info(f"------Stage {Stage_name}------------")
    excel_path = ingest_data(file_url = SOURCE_URL)
    load_data(pickle_path = PICKLE_PATH, excel_path = excel_path)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
except Exception as e:
        logging.error(e)