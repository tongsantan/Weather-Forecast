## 6. Update the pipeline

from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.data_preprocessing_feature_engineering()
        data_ingestion.complete_data_ingestion()

if __name__ == "__main__":
    try:
        logger.info(f">>>>> {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    