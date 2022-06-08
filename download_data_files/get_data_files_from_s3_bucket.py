import os
from utils.utils import read_config
import boto3
from application_logging import logger
class Get_Data:
    def __init__(self,path):
        config=read_config()
        self.Access_key_ID=os.getenv(config['aws_credebtials_environment_variables']['Access_key_ID'])
        self.Secret_access_key=os.getenv(config['aws_credebtials_environment_variables']['Secret_access_key'])
        self.bucket_name=config['feature_store']['bucket_name']
        self.path=path
        self.file_object = open("Training_Logs/get_data_from_s3.txt", 'a+')
        self.log_writer = logger.App_Logger()
    def download_data_files(self):
        self.log_writer.log(self.file_object, f'Connecting to s3 resource bucket_name {self.bucket_name}')
        s3 = boto3.resource('s3',aws_access_key_id=self.Access_key_ID,aws_secret_access_key=self.Secret_access_key)     
        my_bucket = s3.Bucket(self.bucket_name)
        for file in my_bucket.objects.all():
            my_bucket.download_file(file.key,os.path.join(self.path,file.key))
            print('Sucessfully get the file ',file.key,"From s3 bucket name",self.bucket_name)
            self.log_writer.log(self.file_object, 'Sucessfully get the file '+file.key+" From s3 bucket name "+self.bucket_name+"to path "+self.path)
if __name__ == '__main__':
    gd=Get_Data()
    gd.download_data_files()
