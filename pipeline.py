import os
import azureml.core
from azureml.core import Workspace, Run, Experiment, Datastore
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

from azureml.telemetry import set_diagnostics_collection

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence

import pandas as pd

import json

print("SDK Version:", azureml.core.VERSION)


workspace_name = '...'
resource_group = '...'
subscription_id = '...' 
workspace_region = '...'

ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group,
                      location = workspace_region,
                      exist_ok=True)

experiment_name =  'PdM_pipeline' # choose a name for experiment

experiment=Experiment(ws, experiment_name)

set_diagnostics_collection(send_diagnostics=True)

# create AML compute
aml_compute_target = "aml-compute"
try:
    aml_compute = AmlCompute(ws, aml_compute_target)
    print("found existing compute target.")
except:
    print("creating new compute target")
    
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                idle_seconds_before_scaledown=1800,
                                                                min_nodes = 0, 
                                                                max_nodes = 4)    
    aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

cd = CondaDependencies.create(pip_packages=["azureml-train-automl"]) 

# Runconfig
amlcompute_run_config = RunConfiguration(framework="python", conda_dependencies=cd)
amlcompute_run_config.environment.docker.enabled = False
amlcompute_run_config.environment.docker.gpu_support = False
amlcompute_run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
amlcompute_run_config.environment.spark.precache_packages = False

def_data_store = ws.get_default_datastore()

automl_step = PythonScriptStep(name="automl_step",
                                script_name="automl_step.py", 
                                compute_target=aml_compute_target, 
                                source_directory='.', #project_folder,
                                allow_reuse=True,
                                runconfig=amlcompute_run_config)

print("AutoML Training Step created.")


steps = [automl_step]
print("Step lists created")

pipeline = Pipeline(workspace=ws, steps=steps)
print ("Pipeline is built")

pipeline.validate()
print("Pipeline validation complete")

pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Pipeline is submitted for execution")

# Wait until the run finishes.
pipeline_run.wait_for_completion(show_output = False)

# Download aml_config info and output of automl_step
def_data_store.download(target_path='.',
                        prefix='aml_config',
                        show_progress=True,
                        overwrite=True)

def_data_store.download(target_path='.',
                        prefix='outputs',
                        show_progress=True,
                        overwrite=True)


model_fname = 'model.pkl'
model_path = os.path.join("outputs", model_fname)

# Upload the model file explicitly into artifacts (for CI/CD)
pipeline_run.upload_file(name = model_path, path_or_stream = model_path)
print('Uploaded the model {} to experiment {}'.format(model_fname, pipeline_run.experiment.name))
