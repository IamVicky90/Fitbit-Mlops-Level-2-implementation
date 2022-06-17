import shutil
import yaml
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def read_config(path=os.path.join('configuration','config.yaml')):
    with open(path,'r') as f:
        dict=yaml.safe_load(f)
    return dict
def create_important_directories_if_not_exist():
    directories=['Prediction_Batch_files', 'Prediction_Raw_Data_Validation', 'Training_Raw_data_validation', 'Prediction_Database', 'Prediction_Logs', 'TrainingArchiveBadData', 'Training_Logs',  'Training_Database', 'preprocessing_data', 'Training_Raw_files_validated','artifacts',  'DataTransform_Training', 'Prediction_Output_File', 'PredictionArchivedBadData', 'Prediction_FileFromDB', 'Training_FileFromDB', 'models', 'EncoderPickle', 'Prediction_Raw_Files_Validated']
    for directory in directories:
        os.makedirs(directory,exist_ok=True)
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
def copy_models_to_artifacts():
    if os.path.exists(os.path.join(os.getcwd(), 'artifacts')):
        shutil.rmtree('artifacts')
    os.makedirs('artifacts',exist_ok=True)
    models_dir=os.path.join(os.getcwd(),read_config()['trainig_configurations']['models_directory'])
    for folder_name in os.listdir(models_dir):
        for file_name in os.listdir(os.path.join(models_dir,folder_name)):
            shutil.copy(os.path.join(models_dir,folder_name,file_name),os.path.join('artifacts',file_name))
