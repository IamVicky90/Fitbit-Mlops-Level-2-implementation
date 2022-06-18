from monitoring.prometheus import REQUEST_TIME, start_http_server, COUNT_p, COUNT_r, COUNT_w
from aws_model_registry.model_registry import ModelRegistryConnection
from data_preprocessing_service.inference_loader import ObjectLoader
from utils.utils import read_config
from fastapi import FastAPI, Request
import uvicorn
from psutil import process_iter
from signal import SIGTERM # or SIGKILL
from flask import Response
from prediction_Validation_Insertion import pred_validation
from predictFromModel import prediction
app = FastAPI()
models={}


class PrepareEndpoints:
    def __init__(self):
        pass

    def inference_object_loader(self) -> None:
        global models

        registry = ModelRegistryConnection()
        registry.get_package_from_prod()
        loader = ObjectLoader()
        prod_objects = loader.load_objects()
        for i, keys in enumerate(prod_objects.keys()):
            models.update({keys:prod_objects[keys]})
           


@REQUEST_TIME.time()
@app.get('/')
def invoke():
    COUNT_w.inc()
    return {"Response": "Hello world from Model Endpoint"}


@REQUEST_TIME.time()
@app.post('/predict')
async def predict(request: Request):
    query = await request.json()
    # query=[2,2,13162,0.550000011920929,6.05999994277954,0,25,13,328,728]
    if not "full_model" in models.keys():
        print('not full_model')
        ss=models['StandardScaler'].transform([query])
        kmeans=models['KMeans'].predict([query])
        result=models[str(kmeans[0])].predict(ss)
    else:
        print('else full_model')
        ss=models['StandardScaler'].transform([query])
        result=models['full_model'].predict(ss)

    result = {"Result": result.tolist()[0]}
    COUNT_p.inc()
    return result

@REQUEST_TIME.time()
@app.post('/batch_prediction')
def predictRouteClient():
    try:
        config=read_config()
        path = config['predictions_configurations']['predictions_batch_files_path']

        pred_val = pred_validation(path) #object initialization

        pred_val.prediction_validation() #calling the prediction_validation function

        pred = prediction(path) #object initialization

        # predicting for dataset present in database
        path = pred.predictionFromModel()
        return Response("Prediction File created at %s!!!" % path)
        

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@REQUEST_TIME.time()
@app.get('/reload')
def reload():
    executor = PrepareEndpoints()
    executor.inference_object_loader()
    COUNT_r.inc()
    return {"Response": "Updating Model In Prod"}

def stop_already_running_port(port):
    for proc in process_iter():
        for conns in proc.connections(kind='inet'):
            if conns.laddr.port == port:
                proc.send_signal(SIGTERM)
if __name__ == "__main__":
    port=8081
    stop_already_running_port(port)
    executor = PrepareEndpoints()
    executor.inference_object_loader()
    start_http_server(5000)
    uvicorn.run(app, host="localhost", port=port)
