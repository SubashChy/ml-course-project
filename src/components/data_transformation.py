import sys
import os
from dataclasses import dataclass
from pathlib import Path 

# Add the project root folder to sys.path to prevent module import errors
root_folder = Path.cwd()
if str(root_folder) not in sys.path:
    sys.path.insert(0, str(root_folder))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Clean root-level utility and custom tracking imports
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Year', 'Month', 'Day', 'XYZ Card Share %']
            categorical_columns = ['ATM Name', 'Weekday', 'Festival Religion', 'Working Day', 'Holiday Sequence']

            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test data for transformation")

            for df in [train_df, test_df]:
                df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='mixed', dayfirst=True)
                
                df['Year'] = df['Transaction Date'].dt.year
                df['Month'] = df['Transaction Date'].dt.month
                df['Day'] = df['Transaction Date'].dt.day
                df.drop(columns=['Transaction Date'], inplace=True)
                
             
                df['XYZ Card Share %'] = (df['No Of XYZ Card Withdrawals'] / df['No Of Withdrawals']) * 100
                
        
                leakage_col = 'Avg Amt Per Withdrawal'
                if leakage_col in df.columns:
                    df.drop(columns=[leakage_col], inplace=True)

            logging.info("Feature engineering and data leakage prevention complete")

          
            preprocessing_obj = self.get_data_transformer_object()

           
            target_column_name = "Total amount Withdrawn"

       
            input_features_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing transformer to train and test columns")

           
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

           
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving the preprocessing pickle object")

            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    try:
        di = DataIngestion()
        train_path, test_path = di.initiate_data_ingestion()
        
        obj = DataTransformation()
        train_arr, test_arr, preprocessor_file = obj.initiate_data_transformation(
            train_path=train_path, 
            test_path=test_path
        )
        
        print("====================================================")
        print("      DATA TRANSFORMATION COMPLETE SUCCESSFULLY   ")
        print("====================================================")
        print(f"Processed Training Matrix Shape: {train_arr.shape}")
        print(f"Processed Testing Matrix Shape:  {test_arr.shape}")
        print(f"Preprocessor Saved Destination:  {preprocessor_file}")
        print("==================================================")

    except Exception as pipeline_error:
        print(f"Pipeline Execution Failed: {pipeline_error}")
        
        
if __name__ == '__main__':
    try:
        
        di = DataIngestion()
        train_path, test_path = di.initiate_data_ingestion()
        
       
        obj = DataTransformation()
        train_arr, test_arr, preprocessor_file = obj.initiate_data_transformation(
            train_path=train_path, 
            test_path=test_path
        )
        
        print("====================================================")
        print("      DATA TRANSFORMATION COMPLETE SUCCESSFULLY   ")
        print("====================================================")
        print(f"Processed Training Matrix Shape: {train_arr.shape}")
        print(f"Processed Testing Matrix Shape:  {test_arr.shape}")
        print(f"Preprocessor Saved Destination:  {preprocessor_file}")
        print("==================================================")

    except Exception as pipeline_error:
        print(f"Pipeline Execution Failed: {pipeline_error}")