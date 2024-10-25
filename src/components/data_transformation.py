import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
# this library is basically used to create pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
# by using this dataclass we will be able to create this preprocessor pickle file inside the artifacts folder
class DataTransformationConfig:
    # this file path will contain the path of any model/models that we have created and we want to save
    # if we are creating any pipeline pickle file we will name it as preprocessor pickle file
    preprocessor_obj_file_path = os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    # the reason of creating the below function is basically to create all the pickle files which will be responsible for data 
    # transformation like converting categorical to numerical and so on
    def get_data_transformer_object(self):
        '''
        this fucntion is responsible for data transformation based on different types of data
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # creating pipeline and handling missing values
            # this pipeline will run on the training dataset
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            # categorical features
            cat_pipeline = Pipeline(
                steps=[
                    # first step would be to handle missing values
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("categorical columns:{categorical_columns}")
            logging.info("numerical columns:{numerical_columns}")
            # now we will combine both categorical and numerical pipeline
            preprocessor = ColumnTransformer(
                [    
                    # first we have given pipeline name , then what pipeline it is and then we have the columns on which we will be performing this.
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data is completed")
            
            logging.info("obtaining preprocessed object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("saved preprocessing object")
            # this is done to save your object in the form of pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)


