import os, json, datetime, sys, argparse
from azureml.core import Workspace, Environment, Run

from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.environment import Environment

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_folder',
    type=str)
FLAGS, unparsed = parser.parse_known_args()

run = Run.get_context()
ws = run.experiment.workspace

model = Model.register(
    model_path= os.path.join(FLAGS.model_folder,"arima_model.pkl"),
    model_name='arima_model_pipeline',  # this is the name the model is registered as
    description="Time series forecasting model for Adventure Works dataset",
    workspace=ws,
)

myenv = Environment.from_conda_specification(name="arima-env", file_path="./arima-env.yml")
inference_config = InferenceConfig(entry_script="score.py", environment=myenv)

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=3, 
                                               tags={'name':'arima-inference', 'framework': 'statsmodels'},
                                               description='arima inference')

service = Model.deploy(workspace=ws,
                           name='arima-inference', 
                           models=[model], 
                           inference_config=inference_config, 
                           deployment_config=aciconfig, overwrite=True)

service.wait_for_deployment(True)
print('Service Statet:')
print(service.state)
print('Service URI:')
print(service.scoring_uri)

print('Smoke Test:')
step_size=[3]
test_sample = json.dumps({"data": step_size})
test_sample = bytes(test_sample, encoding="utf8")
print(test_sample)

prediction = service.run(input_data=test_sample)
print(prediction)
