#!/usr/bin/env python
# coding: utf-8

# # TODO:
# - [ ] refactor frame creation
# - [ ] save labels and label map in vid folder
# - [ ] alert if vid with same name already exists
# - [ ]  fix logging

# # Setup

# ## imports

# In[1]:


import os
import sys
import time
import json
from shutil import copy
from sklearn.utils import shuffle
import datetime as dt


# In[2]:


from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array


# In[3]:


import cv2


# In[4]:


from contextlib import redirect_stdout


# In[5]:


# setup matplotlib to display plots in the notebook
# %matplotlib inline

# third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# setup display options
pd.options.display.max_rows = 200
pd.options.display.max_colwidth = 400
pd.options.display.float_format = '{:,.5g}'.format
np.set_printoptions(precision=5, suppress=False)

# setup seaborn to use matplotlib defaults & styles
sns.set()
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'axes.grid' : False})


# ## paths

# In[6]:


pwd = os.path.dirname(os.getcwd()) + '/'
pwd


# In[7]:


path_cache = pwd + 'cache/'
path_models = pwd + 'models/'


# In[8]:


# for constructing vids in cache REFACTOR
path_data = pwd + 'data/images/'


# In[9]:


# folder where we'll store each vid grouped into folders
path_vids = path_cache + 'vids/'


# ## setup logging

# In[10]:


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(pwd, "logs")),
        logging.StreamHandler()
    ])

logger = logging.getLogger()


# # helper functions

# In[11]:


def plot_pic(imgpath):
    plt.imshow(plt.imread(imgpath))
    plt.show()


# # Convert data back into sequence for video classification

# ## read list of paths across train/val/test and seal / no seal folders

# In[12]:


path_data


# In[13]:


paths_jpgs = []
for folder, subs, files in os.walk(path_data):        
    for filename in files:
        if filename[-4:] == '.jpg' or  filename[-4:] == 'jpeg':
            paths_jpgs.append(os.path.abspath(os.path.join(folder, filename)))


# In[14]:


# create dataframe from paths
dfp = pd.DataFrame(paths_jpgs)
dfp.columns = ['path']
dfp['label'] = dfp['path'].str.split("/").str.get(-2)
dfp['filename'] = dfp['path'].str.split("/").str.get(-1)
dfp.sort_values("filename", inplace=True)
dfp.reset_index(inplace=True,drop=True)
dfp['vid'] = dfp['filename'].str.split("-").str.get(0) + '-' + dfp['filename'].str.split("-").str.get(1)
dfp['seal'] = pd.get_dummies(dfp['label'])['seal']
dfp.to_csv(path_vids + "df.csv")


# In[15]:


vids = list(dfp['vid'].unique())


# # Create train/test split

# ## functions to load precomputed sequence data for list of vids

# In[16]:


def get_sequence_data_for_vids(list_of_vid_names, sequence_length, pretrained_model_name, pooling):
    """
    Load precomputed sequence features data of given length together with targets [returns data later to train/eval models: x, y]
    for list of vid names and concatenate into one long array
    
    Args:
        list_of_vid_names: name of vid (should already have frames in folder in `/cache/vids/*vid_name*/frames/`)
        sequence_length: length of sequence to fetch precomputed features for
        pretrained_model_name: name of pretrained model whose features should be loaded (assuming these were already precomputed)
        pooling: pooling method used with pretrained model
    
    Returns:
        sequence_features_array_for_all_vids, sequence_targets_array_for_all_vids
    
    """
    
    # create clips of length NUM_FRAMES
    x = []
    y = []
    
    for v, vid_name in enumerate(list_of_vid_names):
        
        path_sequences_features = path_vids + vid_name + '/sequences/features_sequence_' + str(sequence_length) + '_' + pretrained_model_name + '_' + pooling + 'pooling.npy'
        path_sequences_targets = path_vids + vid_name + '/sequences/targets_sequence_' + str(sequence_length) + '.npy'

        # load precomputed features
        features = np.load(path_sequences_features)
        targets = np.load(path_sequences_targets)
        
        x.extend(features)
        y.extend(targets)

    return np.array(x), np.array(y)


# ## generate train / test split

# > todo: write functions to make this quicker

# > todo: cross-validation

# > todo: experiments with config files and json results with function to aggregate to dataframe

# In[207]:


np.random.seed(1337)


# In[208]:


ids = list(range(0,len(vids)))
np.random.shuffle(ids)


# In[210]:


str(ids)


# In[156]:


vids_train = [vids[c] for c in ids[0:40]]
vids_valid = [vids[c] for c in ids[40:44]]
vids_test = [vids[c] for c in ids[44:]]


# In[157]:


vids_valid


# In[158]:


vids_test


# # Fit models

# In[185]:


def fit_model(model_id, architecture, layer_1_sizefactor, layer_2_sizefactor, layer_3_sizefactor, dropout, sequence_length, pretrained_model_name, pooling):

    ###########################
    ### create folder for model 
    ###########################

    path_model = path_models + str(model_id) + '/'
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    ###########################
    ### create train/test split
    ###########################

    x_train, y_train = get_sequence_data_for_vids(vids_train, sequence_length, pretrained_model_name, pooling)
    x_valid, y_valid = get_sequence_data_for_vids(vids_valid, sequence_length, pretrained_model_name, pooling)
    x_test, y_test = get_sequence_data_for_vids(vids_test, sequence_length, pretrained_model_name, pooling)

    # shuffle test and train batches
    x_train, y_train = shuffle(x_train, y_train)
    x_valid, y_valid = shuffle(x_valid, y_valid)

    # CREATE CLASS BALANCE
#     y_train = pd.DataFrame(y_train)
#     keeps = list(y_train[y_train[0]==1].head(int(len(y_train[y_train[0]==1].index)/2)).index)
#     keeps2 = list(y_train[y_train[0] == 0].index)
#     keeps.extend(keeps2)
#     #
#     y_train = y_train.iloc[keeps]
#     y_train = y_train.values
#     x_train = x_train[keeps,:,:]

    NUM_CLASSES = y_train.shape[1]
    NUM_FEATURES = x_train.shape[2]
    SEQ_LENGTH = x_train.shape[1]
    
#     # reshape sequence length 1 into image features if not a sequence
#     if sequence_length == 1:
#         x_train = np.squeeze(x_train, axis=1)
#         x_valid = np.squeeze(x_valid, axis=1)
#         x_test = np.squeeze(x_test, axis=1)



    ##############################
    ### keep track of model params
    ##############################

    # create dict with model parameters
    model_params = {}

    model_params['id'] = str(model_id)

    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    model_params['fit_batch_size'] = BATCH_SIZE
    
    model_params['fit_num_classes'] = NUM_CLASSES
    model_params['model_num_features'] = NUM_FEATURES
    model_params['model_sequence_length'] = SEQ_LENGTH
    
    model_params['pretrained_model_name'] = pretrained_model_name
    model_params['pretrained_model_pooling'] = pooling

    model_params['shape_x_test'] = str(x_train.shape)
    model_params['shape_y_train'] = str(y_train.shape)
    model_params['shape_x_test'] = str(x_test.shape)

    model_params['model_architecture'] = architecture
    model_params['model_layer_1_sizefactor'] = layer_1_sizefactor
    model_params['model_layer_2_sizefactor'] = layer_2_sizefactor
    model_params['model_layer_3_sizefactor'] = layer_3_sizefactor
    model_params['model_dropout'] = dropout

    ################
    ### define model
    ################

    if architecture == "LSTM":
        # https://github.com/sagarvegad/Video-Classification-CNN-and-LSTM-/blob/master/train_CNN_RNN.py
        model = Sequential()

        # layer 1 (LSTM layer)
        model.add(LSTM(NUM_FEATURES//layer_1_sizefactor, return_sequences=False, dropout=dropout, input_shape=(SEQ_LENGTH, NUM_FEATURES)))

        # layer 2 (dense)
        if layer_2_sizefactor > 0:
            model.add(Dropout(dropout))
            model.add(Dense(NUM_FEATURES//layer_2_sizefactor, activation='relu'))

        # layer 3 (dense)
        if layer_2_sizefactor > 0 and layer_3_sizefactor > 0:
            model.add(Dropout(dropout))
            model.add(Dense(NUM_FEATURES//layer_3_sizefactor, activation='relu'))

        # final layer
        model.add(Dropout(dropout))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        # define optimizer and compile model
        opt = Adam()
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if architecture == 'MLP':
        model = Sequential()
        model.add(Flatten(input_shape=(SEQ_LENGTH, NUM_FEATURES)))
        model.add(Dense(NUM_FEATURES//2, activation='relu'))

        if layer_2_sizefactor > 0:
            model.add(Dense(NUM_FEATURES//layer_2_sizefactor, activation='relu'))
            model.add(Dropout(dropout))

        if layer_2_sizefactor > 0 and layer_3_sizefactor > 0:
            model.add(Dense(NUM_FEATURES//layer_3_sizefactor, activation='relu'))
            model.add(Dropout(dropout))

        # final layer
        model.add(Dropout(dropout))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        # define optimizer and compile model
        opt = Adam()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # save model summary to file
    with open(path_model + 'model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # track number of params in model
    model_params['param_count'] = model.count_params()

    #############
    ### fit model
    #############

    # setup training callbacks
    stopper_patience = 10
    model_params['fit_stopper_patience'] = stopper_patience
    callback_stopper = EarlyStopping(monitor='val_acc', patience=stopper_patience, verbose=0)
    callback_csvlogger = CSVLogger(path_model + 'training.log')
    callback_checkpointer = ModelCheckpoint(path_model +  'model.h5', monitor='val_acc', 
                                 save_best_only=True, verbose=0)
    
    start = dt.datetime.now()
    model_params['dt_start'] = start.strftime("%Y-%m-%d %H:%M:%S")

    history = model.fit(x_train, y_train, 
              validation_data=(x_valid,y_valid),
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              callbacks=[callback_stopper, callback_checkpointer, callback_csvlogger],
              shuffle=True,
              verbose=0)
    
    end = dt.datetime.now()    
    model_params['dt_end'] = end.strftime("%Y-%m-%d %H:%M:%S")
    model_params['dt_duration_seconds'] = str((end - start).total_seconds()).split(".")[0]

    # get number of epochs actually trained (might have early stopped)
    epochs_trained = 0
    try:
        epochs_trained = callback_stopper.stopped_epoch - stopper_patience
    except:
        logger.info("WTF")
        pass
    logger.info("XXXXX" + str(epochs_trained))
    model_params['fit_stopped_early'] = True
    if epochs_trained == 0 and len(history.history) > stopper_patience:
        model_params['fit_stopped_early'] = False
        epochs_trained = NUM_EPOCHS - 1 

    model_params['fit_num_epochs'] = epochs_trained
    model_params['fit_val_acc'] = history.history['val_acc'][epochs_trained]
    model_params['fit_train_acc'] = history.history['acc'][epochs_trained]
    model_params['fit_val_loss'] = history.history['val_loss'][epochs_trained]
    model_params['fit_train_loss'] = history.history['loss'][epochs_trained]
    

    #######################
    ### predict on test set
    #######################

    # calculate predictions on test set
    predictions = model.predict(x_test)

    # calculate test error 
    pdf = pd.DataFrame(predictions)
    pdf.columns = ['noseal','seal']

    # get filenames for predictions
    filenames = []
    for vid_name in vids_test:
        filenames_vid = list(dfp[dfp['vid'] == vid_name]['path'])
        filenames.extend(filenames_vid[sequence_length-1:])
    pdf['filename'] = filenames
    print(len(filenames))
    truth = pd.DataFrame(y_test)
    truth.columns = ['truth_noseal','truth_seal']
    truth = truth[['truth_seal']]
    pdf['prediction'] = pdf['seal'].apply(lambda x: round(x))
    pdf = pd.concat([pdf, truth], axis=1)
    pdf['error'] = (pdf['prediction'] != pdf['truth_seal']).astype(int)

    test_acc = 1 - pdf['error'].mean()

    pdf.to_csv(path_model + 'test_predictions.csv')

    model_params['fit_test_acc'] = 1 - pdf['error'].mean()
    logger.info("model {} test acc: {}".format(model_id, test_acc))

    #############################
    ### save model params to file
    #############################
    with open(path_model + 'params.json', 'w') as f:
        json.dump(model_params, f)


# In[35]:


# pretrained_model_names = ["inception_resnet_v2", "inception_v3", "mobilenetv2_1.00_224", "resnet50", "vgg16", "xception"]
# poolings = ['avg','max']
# sequence_lengths = [1, 3, 5, 10, 15, 20, 40]

# architectures = ['LSTM', "MLP"]
# layer_1_sizefactors = [1,2,4,8]
# layer_2_sizefactors = [0,1,2,4,8]
# layer_3_sizefactors = [0,1,2,4,8]
# dropouts = [0, 0.1, 0.2,0.3,0.4,0.5]


# In[188]:


pretrained_model_names = ["inception_resnet_v2"]
poolings = ['avg']
sequence_lengths = [1, 3]

architectures = ["MLP", "LSTM"]
layer_1_sizefactors = [1,2,4,8]
layer_2_sizefactors = [0,2,4,8]
layer_3_sizefactors = [0,2,4,8]
dropouts = [0.2, 0.5]


# In[190]:


experiment_count_total = 0
for pretrained_model_name in pretrained_model_names:
    for pooling in poolings:
        for sequence_length in sequence_lengths:
            for architecture in architectures:
                for layer_1_sizefactor in layer_1_sizefactors:
                    for layer_2_sizefactor in layer_2_sizefactors:
                        for layer_3_sizefactor in layer_3_sizefactors:
                            for dropout in dropouts:
                                experiment_count_total+=1
experiment_count_total


# In[191]:


model_id = 1
experiment_count = 1

for pretrained_model_name in pretrained_model_names:
    for pooling in poolings:
        for sequence_length in sequence_lengths:
            for architecture in architectures:
                for layer_1_sizefactor in layer_1_sizefactors:
                    for layer_2_sizefactor in layer_2_sizefactors:
                        for layer_3_sizefactor in layer_3_sizefactors:
                            for dropout in dropouts:
                                
                                # skip LSTM experiement if not a sequence
                                if sequence_length == 1 and architecture == "LSTM":
                                    continue
                                
                                # log experiment
                                param_names = ["model_id", "architecture", "layer_1_sizefactor", "layer_2_sizefactor", "layer_3_sizefactor", "dropout", "sequence_length", "pretrained_model_name", "pooling"]
                                param_values = [str(x) for x in [model_id, architecture, layer_1_sizefactor, layer_2_sizefactor, layer_3_sizefactor, dropout, sequence_length, pretrained_model_name, pooling]]

                                experiment_description = ""
                                for c, p in enumerate(param_names):
                                    experiment_description += p + ':' + param_values[c] + ', '

                                if not os.path.exists(path_models + str(model_id)):
                                    # run experiment
                                    logging.info("begin experiment {}/{} - {}".format(experiment_count, experiment_count_total, experiment_description))
                                    try:
                                        fit_model(model_id, architecture, layer_1_sizefactor, layer_2_sizefactor, layer_3_sizefactor, dropout, sequence_length, pretrained_model_name, pooling)
                                    except:
                                        logging.info("Error fitting model {}".format(model_id))
                                        pass
                                
                                experiment_count += 1
                                model_id+=1


# # Examine results

# In[186]:


results = []
for folder, subs, files in os.walk(path_models):        
    for filename in files:
        if filename == 'params.json':
            with open(os.path.abspath(os.path.join(folder, filename))) as f:
                data = json.load(f)
            results.append(data)

results = pd.DataFrame(results)

results.sort_values("fit_test_acc", inplace=True, ascending=False)


# In[176]:


results.sort_values("fit_val_acc", inplace=True, ascending=False)


# In[177]:


results.to_csv(pwd+'results.csv')


# # Analyze results of balanced vs unbalanced fits

# In[171]:


# results2 = pd.read_csv(pwd + 'results/results1.csv', index_col=0)

# results['type'] = 'unbalanced'
# results2['type'] = 'balanced'

# results = pd.concat([results,results2],axis=0)

# results.head().T

