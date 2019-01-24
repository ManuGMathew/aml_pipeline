import argparse
import pickle
import json
import numpy as np
from sklearn import datasets

import logging

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.core.run import Run


from azureml.telemetry import set_diagnostics_collection
set_diagnostics_collection(send_diagnostics = True)

digits = datasets.load_digits()

# Exclude the first 100 rows from training so that they can be used for test.
X_train = digits.data[100:,:]
y_train = digits.target[100:]

run = Run.get_context()

ws = run.experiment.workspace
def_data_store = ws.get_default_datastore()

# Choose a name for the experiment and specify the project folder.
experiment_name = 'automl-local-classification'
project_folder = './sample_projects/automl-local-classification'

experiment = Experiment(ws, experiment_name)

primary_metric = 'accuracy'

automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             primary_metric = primary_metric,
                             iteration_timeout_minutes = 60,
                             iterations = 2,
                             n_cross_validations = 3,
                             verbosity = logging.INFO,
                             X = X_train,
                             y = y_train,
                             path = project_folder)

local_run = experiment.submit(automl_config, show_output = True)

# Wait until the run finishes.
local_run.wait_for_completion(show_output = True)

# create new AutoMLRun object to ensure everything is in order
ml_run = AutoMLRun(experiment = experiment, run_id = local_run.id)

# aux function for comparing performance of runs (quick workaround for automl's _get_max_min_comparator)
def maximize(x, y):
    if x >= y:
        return x
    else:
        return y

# next couple of lines are stripped down version of automl's get_output
children = list(ml_run.get_children())

best_run = None # will be child run with best performance
best_score = None # performance of that child run

for child in children:
    candidate_score = child.get_metrics()[primary_metric]
    if not np.isnan(candidate_score):
        if best_score is None:
            best_score = candidate_score
            best_run = child
        else:
            new_score = maximize(best_score, candidate_score)
            if new_score != best_score:
                best_score = new_score
                best_run = child    

# print accuracy                 
best_accuracy = best_run.get_metrics()['accuracy']
print("Best run accuracy:", best_accuracy)

# download model and save to pkl
model_path = "outputs/model.pkl"
best_run.download_file(name=model_path, output_file_path=model_path) 

# Writing the run id to /aml_config/run_id.json
run_id = {}
run_id['run_id'] = best_run.id
run_id['experiment_name'] = best_run.experiment.name

# save run info 
os.makedirs('aml_config', exist_ok=True)
with open('aml_config/run_id.json', 'w') as outfile:
    json.dump(run_id, outfile)

# upload run infor and model (pkl) to def_data_store, so that pipeline mast can access it
def_data_store.upload(src_dir='aml_config', target_path='aml_config', overwrite=True)
def_data_store.upload(src_dir='outputs', target_path='outputs', overwrite=True)
