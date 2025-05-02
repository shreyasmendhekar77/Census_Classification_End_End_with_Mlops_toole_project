# first adding the data-ingestion pipeline
from Mlflow_Ineuron_Project import logger
from Mlflow_Ineuron_Project.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from Mlflow_Ineuron_Project.pipeline.data_cleaning_pipeline import DataPreprocessingPipeline
from Mlflow_Ineuron_Project.pipeline.data_transfromation import DataTransformationPipeline
from Mlflow_Ineuron_Project.pipeline.model_training_pipeline import ModelTrainingPipeline
from Mlflow_Ineuron_Project.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline
import os

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/ShreyasDominator/Mlops_Ineuron_mlflow_census_calssification.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="ShreyasDominator"
os.environ["MLFLOW_TRACKING_PASSWORD"]="c79b27cd042d93400b3a1099115f3899a027693f"

logger.info('Welcome to our custom logging')
STAGE_NAME=" Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME=" Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataPreprocessingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME=" Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Trainer stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model evaluation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelEvaluationTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



