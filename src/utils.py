import os
import sys
from pathlib import Path

# Add project root directory to system paths to fix module imports
root_folder = Path.cwd()
if str(root_folder) not in sys.path:
    sys.path.insert(0, str(root_folder))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """ Simple function to save python objects as pickle files """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Successfully saved pickle artifact to {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """ Runs GridSearch optimization across chosen algorithms and saves performance results """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Perform grid optimization search
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Re-initialize the model structure using the optimal found settings
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Calculate predictions on testing dataset matrices
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            # Assign scoring to matching algorithm key
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    