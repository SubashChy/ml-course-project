import os
import sys
from dataclasses import dataclass
from pathlib import Path

root_folder = Path.cwd()
if str(root_folder) not in sys.path:
    sys.path.insert(0, str(root_folder))

from sklearn.metrics import r2_score

# Baseline algorithms for tournament ranking
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


from src.utils import save_object, evaluate_models
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:

    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data arrays")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            s
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
               
                "CatBoosting Regressor": {},
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }


            logging.info("Evaluating models to find the best configuration")
            
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            
            best_model_score = max(sorted(model_report.values()))

            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            
            best_model = models[best_model_name]

            
            if best_model_score < 0.6:
                raise CustomException("No best model found with an R2 score above 0.60")
                
            logging.info(f"Winner found: {best_model_name} with score {best_model_score:.4f}")

           
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    try:
        import numpy as np
        import pandas as pd
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation
        
        print("==================================================")
        print("          STARTING PIPELINE EXECUTION             ")
        print("==================================================")

       
        train_csv_path = os.path.join('artifacts', 'train.csv')
        test_csv_path = os.path.join('artifacts', 'test.csv')
        preprocessor_pkl_path = os.path.join('artifacts', 'preprocessor.pkl')

        
        train_arr = None
        test_arr = None

     
        if os.path.exists(train_csv_path) and os.path.exists(test_csv_path) and os.path.exists(preprocessor_pkl_path):
            print("[INFO] Found existing artifacts. Loading data from artifacts folder...")
            

            train_df = pd.read_csv(train_csv_path)
            test_df = pd.read_csv(test_csv_path)
            
          
            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(
                train_path=train_csv_path, 
                test_path=test_csv_path
            )
        else:
            print("[INFO] Artifacts missing. Executing full pipeline from scratch...")
            
        
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            print(f"[SUCCESS] Ingestion completed. Train path: {train_path}, Test path: {test_path}")

            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_file = transformation.initiate_data_transformation(
                train_path=train_path, 
                test_path=test_path
            )
            print(f"[SUCCESS] Transformation completed. Preprocessor saved at: {preprocessor_file}")

        print(f"[INFO] Processed Training Matrix Shape: {train_arr.shape}")
        print(f"[INFO] Processed Testing Matrix Shape:  {test_arr.shape}")

        print("[INFO] Initializing Model Trainer tournament...")
        trainer = ModelTrainer()
        final_r2_score = trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)

        print("\n==================================================")
        print("          PIPELINE EXECUTED SUCCESSFULLY          ")
        print("==================================================")
        print(f" Winning Model Final R² Validation Score: {final_r2_score:.4f}")
        print(f" Trained Model Artifact Location:        {trainer.model_trainer_config.trained_model_file_path}")
        print("==================================================\n")

    except Exception as pipeline_error:
        print(f"\n[CRITICAL ERROR] Pipeline execution failed: {pipeline_error}")
