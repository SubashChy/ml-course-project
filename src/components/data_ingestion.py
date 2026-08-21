import os
import sys
from pathlib import Path 
root_folder = Path.cwd()

if str(root_folder) not in sys.path:
    sys.path.insert(0 , str(root_folder))
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    # Where to save the files we create
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")
            
        
            df = pd.read_csv(os.path.join('notebooks/data', 'transactions_in_usd.csv'))
            logging.info("Loaded the CSV file successfully")

            text_columns = ['Weekday', 'ATM Name', 'Festival Religion', 'Working Day', 'Holiday Sequence']
            
            logging.info("Fixing spaces and text casing in columns")
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.title()

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved the raw data file")

            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion process completed completely")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
           
            raise CustomException(e, sys)


if __name__ == '__main__':
    di = DataIngestion()
    di.initiate_data_ingestion()