import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# decorator
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # with the help of this whenever we will call the DataIngestion class all the three paths will get stored in ingestion_config
    # we have created this function to read the data , say if it is stored in databases
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            # here only we can read it from mongo db or from anywhere that we want ie api also
            df= pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the datasets as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            # saving in the artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                # we are returning them so that we can use them in data transformation
            )
        except Exception as e:
            raise CustomException(e,sys)

# if __name__=="__main__":
#     obj=DataIngestion()
#     train_data,test_data=obj.initiate_data_ingestion()

#     data_transformation=DataTransformation()
#     # we are saving train and test arr so that we can forward it to model trainer 
#     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

#     modelTrainer = ModelTrainer()
#     print(modelTrainer.initiate_model_trainer(train_arr,test_arr))

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

