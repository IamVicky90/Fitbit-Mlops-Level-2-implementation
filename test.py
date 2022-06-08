# from sklearn.ensemble import RandomForestClassifier
# rfc=RandomForestClassifier(200)
# print(rfc.n_estimators)
# from sklearn.ensemble import RandomForestRegressor
# import sklearn
# print(sklearn.__version__)
# rfr=RandomForestRegressor()
from utils.utils import read_config, copy_models_to_artifacts

# copy_models_to_artifacts()
from aws_model_registry import model_registry
model_registry.ModelRegistryConnection().upload_model_in_test()
model_registry.ModelRegistryConnection().get_package_from_testing()
model_registry.ModelRegistryConnection().move_model_test_to_prod()
import os, glob
print(glob.glob(os.path.join("artifacts","*.sav")))
# print(glob.glob("artifacts"))

