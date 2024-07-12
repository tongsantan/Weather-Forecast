## 6. Update the pipeline

from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation
from src import logger

STAGE_NAME = "Data Transformation"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.get_data_transformer_object()
        data_transformation.initiate_data_transformation()


if __name__ == "__main__":
    try:
        logger.info(f"********************************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e