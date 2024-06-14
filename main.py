from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

from src import logger

if __name__ == '__main__':
    # data ingestion pipeline
    STAGE_NAME = "Data Ingestion stage"
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    # data transformation pipeline
    STAGE_NAME = "Data Transformation stage"
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    # model training pipeline
    STAGE_NAME = "Model Training stage"
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    modeltrainer=ModelTrainer()
    print("Best Model: {}, Best F1 Score: {}, Confusion Matrix: {}".format(*modeltrainer.initiate_model_trainer(train_arr,test_arr)))
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")