from utils.utils import read_config
from aws_model_registry.model_registry import ModelRegistryConnection
from from_root import from_root
from pickle import load
import os
import glob

class ObjectLoader:
    def __init__(self):
        self.config = read_config()
        self.extract_folder = os.path.join(from_root(), "artifacts")

    def load_objects(self):
        objects = {}
        
        for file_path in glob.glob(os.path.join("artifacts","*.sav")):
                if 'KMeans' in file_path or 'StandardScaler' in file_path:
                    with open(file_path,'rb') as f:
                        objects[file_path.split("/")[1].replace(".sav", "")] = load(f)
                elif "full_model" in file_path:
                    with open(file_path,'rb') as f:
                        objects["full_model"] = load(f)
                elif not "full_model" in ''.join([str(s) for s in glob.glob(os.path.join("artifacts","*.sav"))]):
                    with open(file_path,'rb') as f:
                        objects[file_path.split("/")[1].replace(".sav", "")[-1]] = load(f)
                os.remove(file_path)

        return objects

