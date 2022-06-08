from  download_data_files.get_data_files_from_s3_bucket import Get_Data
from aws_model_registry.model_registry import ModelRegistryConnection
from data_preprocessing_service.inference_loader import ObjectLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils.utils import read_config
import requests
import sys
from training_Validation_Insertion import train_validation
from application_logging import logger
from data_preprocessing import preprocessing
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
import statistics
import math
import pandas
class ModelTest:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTestingLog.txt", 'a+')
        self.config=read_config()
        # self.model_endpoint = "https://a00f2d17e0d0.in.ngrok.io/reload"
        self.model_endpoint = " http://localhost:8081/reload"

    

    def additional_preprocess(self, raw_Data):
        preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
        raw_Data = preprocessor.dropUnnecessaryColumns(raw_Data,self.config['trainig_configurations']['dropUnnecessaryColumns'])
        X, Y = preprocessor.separate_label_feature(raw_Data, label_column_name=self.config['trainig_configurations']['label_column_name'])
        _, X_test, _, y_test = train_test_split(X, Y, test_size=self.config['trainig_configurations']['train_test_split']['test_size'], random_state=self.config['trainig_configurations']['train_test_split']['random_state'])
        return X_test, y_test

    @staticmethod
    def get_predictions(objects, X_test, y_test):
        models={}
        if not "full_model" in str(objects.keys()):
            r2=[]
            mae=[]
            rmse=[]
            for i, keys in enumerate(objects.keys()):
                models.update({keys:objects[keys]})
            ss=models['StandardScaler'].transform(X_test)
            kmeans=models['KMeans'].predict(ss)
            X_test['Cluster']=kmeans
            X_test['Labels']=y_test
            
            print("Prediction started")
            for i in X_test['Cluster'].unique():
                cluster_data=X_test[X_test['Cluster']==i] # filter the data for one cluster
                    # Prepare the feature and Label columns
                cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                cluster_label= cluster_data['Labels']
                ss=models['StandardScaler'].transform(cluster_features)
                prediction=models[str(kmeans[i])].predict(pandas.DataFrame(ss,columns=cluster_features.columns))
                # prediction=models[str(kmeans[i])].predict(ss)
                r2.append(r2_score(cluster_label, prediction))
                mae.append(mean_absolute_error(cluster_label, prediction))
                rmse.append(math.sqrt(mean_squared_error(cluster_label, prediction)))
            print(r2)
            return statistics.mean(r2), statistics.mean(mae), statistics.mean(rmse)
        else:
            for i, keys in enumerate(objects.keys()):
                models.update({keys:objects[keys]})
            ss=models['StandardScaler'].transform(X_test)
            prediction=models['full_model'].predict(pandas.DataFrame(ss,columns=X_test.columns))
            r2=r2_score(y_test, prediction)
            mae=mean_absolute_error(y_test, prediction)
            rmse=math.sqrt(mean_squared_error(y_test, prediction))
            return r2,mae,rmse

    def test(self):
            config=read_config()
            path=config['trainig_configurations']['training_batch_files_path']
            get_data=Get_Data(path)
            get_data.download_data_files()
            # path="Training_Batch_Files"
            train_valObj = train_validation(path) #object initialization

            train_valObj.train_validation()#calling the training_validation function
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            raw_data=data_getter.get_data()
            X_test, y_test=self.additional_preprocess(raw_data)
            registry=ModelRegistryConnection()
            loader = ObjectLoader()
            registry.get_package_from_testing()
            test_objects = loader.load_objects()

            r2_test, _, _ = self.get_predictions(test_objects, X_test, y_test)
            print(f"Testing objects loaded {test_objects}")
            self.log_writer.log(self.file_object,f"Testing objects loaded {test_objects}")

            X_test, y_test=self.additional_preprocess(raw_data)
            registry.get_package_from_prod()
            prod_objects = loader.load_objects()
            r2_prod, _, _ = self.get_predictions(prod_objects, X_test, y_test)
            print(f"Production objects loaded {prod_objects}")
            self.log_writer.log(self.file_object,f"Production objects loaded {prod_objects}")

            print("checking condition")
            self.log_writer.log(self.file_object,"checking condition")
            print(f"F1 Score Test {r2_test}")
            self.log_writer.log(self.file_object,f"F1 Score Test {r2_test}")
            print(f"F1 Score Prod {r2_prod}")
            self.log_writer.log(self.file_object,f"F1 Score Prod {r2_prod}")

            if r2_test > r2_prod:
                self.log_writer.log(self.file_object,f"r2_test > r2_prod so moving to prod")
                response = registry.move_model_test_to_prod()
                # reload = requests.get(self.model_endpoint)
                # print(reload.text)
                print(response)
            else:
                self.log_writer.log(self.file_object,f"r2_test < r2_prod so, Prod model is More accurate")
                print("Prod model is More accurate")

            return True
        


if __name__ == "__main__":
    model_selector = ModelTest()
    result = model_selector.test()
    print(result)
