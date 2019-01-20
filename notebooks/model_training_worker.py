#!/usr/bin/env python
# coding: utf-8


# WORKER_ID="1"
# experiment_batch_name="experiment_batch_1"
# GPU_ID="1"

"""
Worker to train models for experiments defined via notebooks/experiments_create_lists.ipynb

Usage:
    python model_training_worker.py --experiment_batch_name=experiment_batch_1 --WORKER_ID=1 --GPU_ID=1
"""


# Setup command line args
import tensorflow as tf
flags = tf.app.flags
flags.DEFINE_string('experiment_batch_name', '', 'Name of experiment batch (corresponds to CSV in /experiments/) with list of experiments and worker assignments - created using /notebooks/experiments_create_lists.ipynb')
flags.DEFINE_string('WORKER_ID', '', 'ID of this worker - assumed that this worker script will be started one time each with a unique workerid such that the total number of workers matches WORKER_COUNT set when creating experiment batch csv file')
flags.DEFINE_string('GPU_ID', '', 'ID of GPU to be used by this worker- same as WORKER_ID if left blank - useful for multiple instance parallelism')
#
experiment_batch_name = flags.FLAGS.experiment_batch_name
WORKER_ID = flags.FLAGS.WORKER_ID
GPU_ID = flags.FLAGS.WORKER_ID
if GPU_ID is None or GPU_ID=="":
    GPU_ID = WORKER_ID
#
print("experiment_batch_name", experiment_batch_name)
print("WORKER_ID", WORKER_ID)
print("GPU_ID", GPU_ID)


# assign GPU for this worker
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_ID)


import gc
import os
import pandas as pd
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
import itertools
import sys
sys.path.append('..')


# setup paths
pwd = os.getcwd().replace("notebooks","")
path_cache = pwd + 'cache/'
path_data = pwd + 'data/'


# setup logging
# separate log file for each worker
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s, [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(pwd, "logs_" + str(WORKER_ID))),
        logging.StreamHandler()
    ])
# init logger - will pass this to our architecture
logger = logging.getLogger()

logger.info("Start worker {} (GPU={}) processing {}".format(WORKER_ID, GPU_ID, experiment_batch_name))

from deepvideoclassification.architectures import Architecture


# # Run experiments

# load list of experiments
experiments = pd.read_csv(pwd + "experiments/" + experiment_batch_name + '.csv')



###################
### Run experiments 
###################

# experiment already done (and therefore skipped) if results.json output already exists
# if WORKER_ID is specified, only those experiments with a matching WORKER_ID column will be run by this worker

for row in experiments.values:
    
    # get experiment params from dataframe row
    experiment = dict(zip(experiments.columns, row))
    
    # only run experiment if not already run
    if not os.path.exists(pwd + 'models/' + str(experiment["model_id"]) + '/results.json'):

        # only run experiment if matches this worker id
        if experiment['WORKER'] == int(WORKER_ID):
            
            logger.info(str(experiment["model_id"]) + "   " + "X"*60)
            logger.info("Begin experiment for model_id={} on GPU:{} ".format(experiment['model_id'], os.environ["CUDA_VISIBLE_DEVICES"]))

            # Define model
            architecture = Architecture(model_id = experiment['model_id'], 
                                        architecture = experiment['architecture'], 
                                        sequence_length = experiment['sequence_length'], 
                                        pretrained_model_name = experiment['pretrained_model_name'],
                                        pooling = experiment['pooling'],
                                        sequence_model = experiment['sequence_model'],
                                        sequence_model_layers = experiment['sequence_model_layers'],
                                        layer_1_size = experiment['layer_1_size'],
                                        layer_2_size = experiment['layer_2_size'],
                                        layer_3_size = experiment['layer_3_size'],
                                        dropout = experiment['dropout'],
                                        verbose=True, 
                                        logger = logger)

            # Train model
            architecture.train_model()
            
            # collect garbage for good luck
            gc.collect()

