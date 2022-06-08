from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from utils.utils import read_config
from utils.utils import eval_metrics
import mlflow
import sys
import os
class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: vicky
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rfr = RandomForestRegressor()
        self.DecisionTreeReg = DecisionTreeRegressor()
        self.config=read_config()
    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: vicky
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = self.config['trainig_configurations']['random_forest']['param_grid']
            # self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
            #                    "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
            # self.param_grid.update({"max_depth": [4]})
            self.param_grid.update({"max_depth": range(2, 4, 1)})

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.rfr, param_grid=self.param_grid, cv=self.config['trainig_configurations']['GridSearchCV']['cv'],  verbose=self.config['trainig_configurations']['GridSearchCV']['verbose'],scoring='r2')
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.bootstrap = self.grid.best_params_['bootstrap']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']

            #creating a new model with the best parameters
            self.rfr = RandomForestRegressor(n_estimators=self.n_estimators,
                                              max_depth=self.max_depth, max_features=self.max_features,bootstrap=self.bootstrap,min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf)
            # training the mew model
            self.rfr.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.rfr
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()
    def get_best_params_for_DecisionTreeRegressor(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_DecisionTreeRegressor
                                                Description: get the parameters for DecisionTreeRegressor Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: vicky
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_DecisionTreeRegressor method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_decisionTree = self.config['trainig_configurations']['decision_tree']['param_grid']
            # self.param_grid_decisionTree = {"criterion": ["mse", "friedman_mse", "mae"],
            #                   "splitter": ["best", "random"],
            #                   "max_features": ["auto", "sqrt", "log2"],
            #                   'max_depth': range(2, 16, 2),
            #                   'min_samples_split': range(2, 16, 2)
            #                   }
            # self.param_grid_decisionTree.update({
            #                   'max_depth': range(2, 4, 2),
            #                   'min_samples_split': range(2, 4, 2)
            #                   })
            self.param_grid_decisionTree.update({
                              'max_depth': range(2, 16, 2),
                              'min_samples_split': range(2, 16, 2)
                              })

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.DecisionTreeReg, self.param_grid_decisionTree,  cv=self.config['trainig_configurations']['GridSearchCV']['cv'],  verbose=self.config['trainig_configurations']['GridSearchCV']['verbose'],scoring='r2')
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.splitter = self.grid.best_params_['splitter']
            self.max_features = self.grid.best_params_['max_features']
            # self.max_depth  = range(2, 16, 2)
            self.max_depth  = self.grid.best_params_['max_depth']
            # self.min_samples_split = range(2, 16, 2)
            self.min_samples_split = self.grid.best_params_['min_samples_split']

            # creating a new model with the best parameters
            self.decisionTreeReg = DecisionTreeRegressor(criterion=self.criterion,splitter=self.splitter,max_features=self.max_features,max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            # training the mew models
            self.decisionTreeReg.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'KNN best params: ' + str(
                                       self.grid.best_params_) + '. Exited the KNN method of the Model_Finder class')
            return self.decisionTreeReg
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in knn method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'knn Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: Vicky
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = self.config['trainig_configurations']['xg_boost']['param_grid']
            # self.param_grid_xgboost = {

            #     'learning_rate': [0.5, 0.1, 0.01, 0.001],
            #     'max_depth': [3, 5, 10, 20],
            #     'n_estimators': [10, 50, 100, 200]

            # }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBRegressor(objective='reg:linear'),self.param_grid_xgboost,  cv=self.config['trainig_configurations']['GridSearchCV']['cv'],  verbose=self.config['trainig_configurations']['GridSearchCV']['verbose'],scoring='r2')
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBRegressor(objective='reg:linear',learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y,i):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: vicky
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for KNN
        try:

            # create best model for RandomForest
            self.random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            # self.random_forest = RandomForestRegressor().fit(train_x, train_y)
            self.prediction_random_forest = self.random_forest.predict(test_x)  # Predictions using the Random Forest Model
            with mlflow.start_run():
                (rmse, mae, r2) = eval_metrics(test_y, self.prediction_random_forest)
                print(f"Cluster {i} random_forest model: n_estimators={self.random_forest.n_estimators}, criterion={self.random_forest.criterion},max_features={self.random_forest.max_features},max_depth={self.random_forest.max_depth},bootstrap={self.random_forest.bootstrap},min_samples_split={self.random_forest.min_samples_split},min_samples_leaf={self.random_forest.min_samples_leaf}",sep="\n")
                print(f"Cluster {i} random_forest RMSE:",rmse)
                print(f"Cluster {i}random_forest MAE:", mae)
                print(f"Cluster {i} random_forest  R2:",r2)
                mlflow.log_param(f"n_estimators", self.random_forest.n_estimators)
                mlflow.log_param(f"criterion", self.random_forest.criterion)
                mlflow.log_param(f"max_features", self.random_forest.max_features)
                mlflow.log_param(f"max_depth", self.random_forest.max_depth)
                mlflow.log_param(f"bootstrap", self.random_forest.bootstrap)
                mlflow.log_param(f"min_samples_split", self.random_forest.min_samples_split)
                mlflow.log_param(f"min_samples_leaf", self.random_forest.min_samples_leaf)
                mlflow.log_metric(f"rmse", rmse)
                mlflow.log_metric(f"r2", r2)
                mlflow.log_metric(f"mae", mae)
                mlflow.sklearn.log_model(self.random_forest, "RandomForestRegressorModel")
            self.prediction_random_forest_r2_score = r2

            # create best model for RandomForest
            self.decisionTreeReg= self.get_best_params_for_DecisionTreeRegressor(train_x, train_y)
            # self.decisionTreeReg= DecisionTreeRegressor().fit(train_x, train_y)
            self.prediction_decisionTreeReg = self.decisionTreeReg.predict(test_x) # Predictions using the decisionTreeReg Model
            with mlflow.start_run():
                (rmse, mae, r2) = eval_metrics(test_y, self.prediction_decisionTreeReg)
                print(f"Cluster {i} Decision Tree model: criterion={self.decisionTreeReg.criterion}, splitter={self.decisionTreeReg.splitter},max_features={self.decisionTreeReg.max_features},max_depth={self.decisionTreeReg.max_depth},min_samples_split={self.decisionTreeReg.min_samples_split}")
                print(f"Cluster {i} Decision Tree RMSE:",rmse)
                print(f"Cluster {i}Decision Tree MAE:" , mae)
                print(f"Cluster {i} Decision Tree  R2:",r2)
                mlflow.log_param(f"criterion", self.decisionTreeReg.criterion)
                mlflow.log_param(f"splitter", self.decisionTreeReg.splitter)
                mlflow.log_param(f"max_features", self.decisionTreeReg.max_features)
                mlflow.log_param(f"max_depth", self.decisionTreeReg.max_depth)
                mlflow.log_param(f"min_samples_split", self.decisionTreeReg.min_samples_split)
                mlflow.log_metric(f"rmse", rmse)
                mlflow.log_metric(f"r2", r2)
                mlflow.log_metric(f"mae", mae)
                mlflow.sklearn.log_model(self.decisionTreeReg, "DecisionTreeRegressorModel")
            self.decisionTreeReg_r2_score = r2



         # create best model for XGBoost
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            # self.xgboost = XGBRegressor().fit(train_x, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x)  # Predictions using the XGBoost Model
            with mlflow.start_run():
                (rmse, mae, r2) = eval_metrics(test_y, self.prediction_xgboost)
                print(f"Cluster {i} xgboost model: learning_rate={self.xgboost.learning_rate}, max_depth={self.xgboost.max_depth},n_estimators={self.xgboost.n_estimators}")
                print(f"Cluster {i} xgboost RMSE:",rmse)
                print(f"Cluster {i}xgboost MAE:",mae)
                print(f"Cluster {i} xgboost  R2:",r2)
                mlflow.log_param(f"learning_rate", self.xgboost.learning_rate)
                mlflow.log_param(f"max_depth", self.xgboost.max_depth)
                mlflow.log_param(f"n_estimators", self.xgboost.n_estimators)
                mlflow.log_metric(f"rmse", rmse)
                mlflow.log_metric(f"r2", r2)
                mlflow.log_metric(f"mae", mae)
                mlflow.xgboost.log_model(self.xgboost, "XgboostRegressorModel")
            self.prediction_xgboost_r2_score = r2
            

         


            #comparing theses three models
            self.r2_scores_of_every_model =[self.decisionTreeReg_r2_score,self.prediction_xgboost_r2_score,self.prediction_random_forest_r2_score]
            print("r2 scores ",self.r2_scores_of_every_model)
            max_value_index=max(self.r2_scores_of_every_model)
            if self.r2_scores_of_every_model.index(max_value_index)==0:
                return 'DecisionTreeReg',self.decisionTreeReg,max(self.r2_scores_of_every_model)
            elif self.r2_scores_of_every_model.index(max_value_index)==1:
                return 'XGBoost',self.xgboost,max(self.r2_scores_of_every_model)
            else:
                return 'Random_Forest',self.random_forest,max(self.r2_scores_of_every_model)


        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            raise Exception()




