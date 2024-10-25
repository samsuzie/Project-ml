import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
# this will give the input whatever we required in terms of model trainer
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # now this function will start the model training
    # in bracket we are giving the output of data transformation
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splititng training and test  input data")
            X_train,y_train,X_test,y_test = (
                # the below index means take out the last column and feed everything to X_train
                # converting the data into features and target for training data and features and target into testing data
                train_array[:,:-1],
                # only taking all the values of last column
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # creating a dictionary of all models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,y_test = y_test,models=models)

            # to get best model score from dictonary
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # putting a threshold
            if best_model_score<0.6:
                raise CustomException("No Best model found")
            logging.info(f"best found model on both training and testing dataset")
            # now we are goinf to save the model path using our save_object functon
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)