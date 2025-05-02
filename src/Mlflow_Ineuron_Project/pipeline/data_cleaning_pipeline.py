from Mlflow_Ineuron_Project.config.configuration import ConfigurationManager
from Mlflow_Ineuron_Project.components.data_preprocessing_and_Cleaning import Data_preprocessing_Validation
from Mlflow_Ineuron_Project import logger

STAGE_NAME="Data cleaning and Preprocessing Stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation =Data_preprocessing_Validation(config=data_validation_config)
        data_validation.validate_all_columns()
    

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
