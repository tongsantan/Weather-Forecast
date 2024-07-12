import os
from pathlib import Path
from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig, 
                                      DataTransformationConfig,
                                      ModelTrainerConfig)

## 4. Update the configuration manager in src config

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH):
        
        self.config = read_yaml(config_filepath)

        create_directories([self.config.output_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            input_data_path=config.input_data_path,
            processed_data_path=config.processed_data_path,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            raw_data_path=config.raw_data_path 
        )

        return data_ingestion_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            train_array_path=config.train_array_path,
            test_array_path=config.test_array_path,
            preprocessor_obj_file_path=config.preprocessor_obj_file_path
        )

        return data_transformation_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            trained_model_file_path=config.trained_model_file_path,
            train_array_path=config.train_array_path,
            test_array_path=config.test_array_path
        )

        return model_trainer_config