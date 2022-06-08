from aws_model_registry import model_registry
from flask import Response
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
from  download_data_files.get_data_files_from_s3_bucket import Get_Data
from utils.utils import read_config,copy_models_to_artifacts




def start_train():

    try:
            config=read_config()
            path=config['trainig_configurations']['training_batch_files_path']
            get_data=Get_Data(path)
            get_data.download_data_files()
            # path="Training_Batch_Files"
            train_valObj = train_validation(path) #object initialization

            train_valObj.train_validation()#calling the training_validation function


            trainModelObj = trainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table
            copy_models_to_artifacts()
            model_registry.ModelRegistryConnection().upload_model_in_test()


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")
if __name__ == "__main__":
    start_train() # Triggers the training

