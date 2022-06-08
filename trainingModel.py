
"""
This is the Entry point for Training the Machine Learning Model.

Written By: Vicky
Version: 1.0
Revisions: None

"""


# Doing the necessary imports
import shutil
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
from utils.utils import read_config
import pandas as pd
from statistics import mean
class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
        self.config=read_config()
    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()


            """doing the data preprocessing"""

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            #data=preprocessor.remove_columns(data,['Wafer']) # remove the unnamed column as it doesn't contribute to prediction.

            #removing unwanted columns as discussed in the EDA part in ipynb file
            data = preprocessor.dropUnnecessaryColumns(data,self.config['trainig_configurations']['dropUnnecessaryColumns'])

            #replacing 'na' values with np.nan as discussed in the EDA part

            data = preprocessor.replaceInvalidValuesWithNull(data)



            # check if missing values are present in the dataset
            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if(is_null_present):
                data=preprocessor.impute_missing_values(data) # missing value imputation

            # get encoded values for categorical data

            #data = preprocessor.encodeCategoricalValues(data)

            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name=self.config['trainig_configurations']['label_column_name'])
            try:
                shutil.rmtree(self.config['trainig_configurations']['models_directory'])
            except Exception as e:
                self.log_writer.log(self.file_object, f"Warning Could not removed models directory from path {self.config['trainig_configurations']['models_directory']} error: {str(e)}")
            X_scaled=preprocessor.standardScalingData(X)
            X=pd.DataFrame(X_scaled,columns=X.columns)
            # drop the columns obtained above
            #X=preprocessor.remove_columns(X,cols_to_drop)


            """ Applying the clustering approach"""

            kmeans=clustering.KMeansClustering(self.file_object,self.log_writer) # object initialization.
            number_of_clusters=kmeans.elbow_plot(X)  #  using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X=kmeans.create_clusters(X,number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels']=Y

            # getting the unique clusters from our dataset
            list_of_clusters=X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""
            r2_score_lst=[]

            for i in list_of_clusters:
                cluster_data=X[X['Cluster']==i] # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                cluster_label= cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=self.config['trainig_configurations']['train_test_split']['test_size'], random_state=self.config['trainig_configurations']['train_test_split']['random_state'])

                # x_train_scaled = preprocessor.standardScalingData(x_train)
                # x_test_scaled = preprocessor.standardScalingData(x_test)

                model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization

                #getting the best model for each of the clusters
                best_model_name,best_model,r2_score=model_finder.get_best_model(x_train,y_train,x_test,y_test,i)
                r2_score_lst.append(r2_score)

                #saving the best model to the directory.
                file_op = file_methods.File_Operation(self.file_object,self.log_writer)
                save_model=file_op.save_model(best_model,best_model_name+str(i))

            #saving the best model to the directory.
            x_train, x_test, y_train, y_test = train_test_split(X.drop(['Labels','Cluster'],axis=1), Y, test_size=self.config['trainig_configurations']['train_test_split']['test_size'], random_state=self.config['trainig_configurations']['train_test_split']['random_state'])

            #getting the best model for each of the clusters
            best_model_name,best_model,r2_score_of_full_model=model_finder.get_best_model(x_train,y_train,x_test,y_test,"full_model")
            print('r2_score_of_full_model',r2_score_of_full_model)
            self.log_writer.log(self.file_object, 'r2_score_of_full_model: '+str(r2_score_of_full_model))
            print('mean(r2_score_lst)',mean(r2_score_lst))
            self.log_writer.log(self.file_object, 'mean(r2_score_lst): '+str(mean(r2_score_lst)))
            print('r2_score_lst',r2_score_lst)
            self.log_writer.log(self.file_object, 'r2_score_lst: '+str(r2_score_lst))
            if r2_score_of_full_model>mean(r2_score_lst):
                self.log_writer.log(self.file_object, 'r2_score_of_full_model>mean(r2_score_lst)')
                self.log_writer.log(self.file_object, 'Saving full_model file.')
                save_model=file_op.save_model(best_model,best_model_name+"full_model")
            self.log_writer.log(self.file_object, 'r2_score_of_full_model<max(r2_score_lst)')
            self.log_writer.log(self.file_object, 'full_model rejected')

            
            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training'+str(e))
            self.file_object.close()
            raise Exception


