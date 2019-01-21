#!/usr/bin/env python
# coding: utf-8


import os
import sys
import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import gc
import itertools
from shutil import copyfile
# from contextlib import redirect_stdout
sys.path.append('..')


# In[5]:


from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, Convolution1D, Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop


# In[6]:


from sklearn.metrics import confusion_matrix


# In[8]:


# setup paths
# pwd = os.getcwd().replace("deepvideoclassification","")
pwd = os.getcwd().replace("notebooks","")
path_cache = pwd + 'cache/'
path_data = pwd + 'data/'

# In[9]:


# In[10]:


from deepvideoclassification.data import Data

# load preprocessing functions
from deepvideoclassification.pretrained_CNNs import load_pretrained_model, load_pretrained_model_preprocessor, precompute_CNN_features
# load preprocessing constants
from deepvideoclassification.pretrained_CNNs import pretrained_model_len_features, pretrained_model_sizes


# # Confusion Matrix

# In[20]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# # Architecture class (contains keras model object and train/evaluate method, writes training results to /models/)

# In[21]:


class Architecture(object):
    
    def __init__(self, model_id, architecture, sequence_length, 
                frame_size = None, 
                pretrained_model_name = None, pooling = None,
                sequence_model = None, sequence_model_layers = None,
                layer_1_size = 0, layer_2_size = 0, layer_3_size = 0, 
                dropout = 0, convolution_kernel_size = 3, 
                model_weights_path = None, 
                batch_size = 32, 
                verbose = False,
                logger = None):
        """
        Model object constructor. Contains Keras model object and training/evaluation methods. Writes model results to /models/_id_ folder
        
        Architecture can be one of: 
        image_MLP_frozen, image_MLP_trainable, video_MLP_concat, video_LRCNN_frozen, video_LRCNN_trainable, C3D, C3Dsmall
        
        :model_id: integer identifier for this model e.g. 1337
        
        :architecture: architecture of model in [image_MLP_frozen, image_MLP_trainable, video_MLP_concat, video_LRCNN_frozen, video_LRCNN_trainable, C3D, C3Dsmall]
        
        :sequence_length: number of frames in sequence to be returned by Data object
        
        :frame_size: size that frames are resized to (different models / architectures accept different input sizes - will be inferred if pretrained_model_name is given since they have fixed sizes)
        :pretrained_model_name: name of pretrained model (or None if not using pretrained model e.g. for 3D-CNN)
        :pooling: name of pooling variant (or None if not using pretrained model e.g. for 3D-CNN or if fitting more non-dense layers on top of pretrained model base)
        
        :sequence_model: sequence model in [LSTM, SimpleRNN, GRU, Convolution1D]
        :sequence_model_layers: default to 1, can be stacked 2 or 3 (but less than 4) layer sequence model (assume always stacking the same sequence model, not mixing LSTM and GRU, for example)
        
        :layer_1_size: number of neurons in layer 1
        :layer_2_size: number of neurons in layer 2
        :layer_3_size: number of neurons in layer 3 
        
        :dropout: amount of dropout to add (same applied throughout model - good default is 0.2) 
        
        :convolution_kernel_size: size of 1-D convolutional kernel for 1-d conv sequence models (good default is 3)
        
        :model_weights_path: path to .h5 weights file to be loaded for pretrained CNN in LRCNN-trainable 
        
        :batch_size: batch size used to fit model (default to 32)
        
        :verbose: whether to log progress updates
        :logger: logger object
        """
    
        # required params
        self.model_id = model_id
        
        self.architecture = architecture
        self.sequence_length = sequence_length
        
        # model architecture params
        self.frame_size = frame_size
        self.pretrained_model_name = pretrained_model_name
        self.pooling = pooling
        self.sequence_model = sequence_model
        self.sequence_model_layers = sequence_model_layers
        #
        self.layer_1_size = layer_1_size
        self.layer_2_size = layer_2_size
        self.layer_3_size = layer_3_size
        #
        self.dropout = dropout
        #
        self.convolution_kernel_size = convolution_kernel_size
        #
        self.model_weights_path = model_weights_path
        #
        self.batch_size = batch_size
        #
        self.verbose = verbose
        
        # fix case sensitivity
        if type(self.architecture) == str:
            self.architecture = self.architecture.lower()
        if type(self.pretrained_model_name) == str:
            self.pretrained_model_name = self.pretrained_model_name.lower()
        if type(self.pooling) == str:
            self.pooling = self.pooling.lower()
        
        # read num features from pretrained model
        if pretrained_model_name is not None:
            self.num_features = pretrained_model_len_features[pretrained_model_name]
            self.frame_size = pretrained_model_sizes[pretrained_model_name]
        
        # check one of pretrained model and frame size is specified
        assert (self.pretrained_model_name is not None or self.frame_size is not None), "Must specify one of pretrained_model_name or frame_size"
            
            
        # init new logger if one was not passed to the class
        self.logger = logger
        if self.logger is None:
            # setup logging
            import logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s, [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
                handlers=[
                    logging.FileHandler("{0}/{1}.log".format(pwd, "logs")),
                    logging.StreamHandler()
                ])
            # init logger - will pass this to our architecture
            self.logger = logging.getLogger()
            
            
        # create path based on model id
        self.path_model = pwd + 'models/' + str(model_id) + '/'
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)
        else:
            if not os.path.exists(self.path_model + 'results.json'):
                self.logger.info("Model folder exists but no results found - potential error in previous model training")
            
        # init model and data objects for this architecture
        self.model = None
        self.data = None
        
        
        #############################################################
        ### Build model architecture and init appropriate data object
        #############################################################
        
        if architecture == "image_MLP_frozen":
            
            ####################
            ### image_MLP_frozen
            ####################
            # image classification (single frame)
            # train MLP on top of weights extracted from pretrained CNN with no fine-tuning
            
            # check inputs
            assert self.sequence_length == 1, "image_MLP_frozen requires sequence length of 1"
            assert self.pretrained_model_name is not None, "image_MLP_frozen requires a pretrained_model_name input" 
            assert self.pooling is not None, "image_MLP_frozen requires a pooling input" 
            
            
            ### create data object for this architecture
            if self.verbose:
                self.logger.info("Loading data")
            self.data = Data(sequence_length = 1, 
                                return_CNN_features = True, 
                                pretrained_model_name= self.pretrained_model_name,
                                pooling = self.pooling)
            
            # init model
            model = Sequential()

            # 1st layer group
            if self.layer_1_size > 0:
                model.add(Dense(self.layer_1_size, activation='relu', input_shape=(self.num_features,)))
                if self.dropout > 0:
                    model.add(Dropout(self.dropout))
                
            # 2nd layer group
            if self.layer_2_size > 0 and self.layer_1_size > 0:
                model.add(Dense(self.layer_2_size, activation='relu'))
                if self.dropout > 0:
                    model.add(Dropout(self.dropout))

            # 3rd layer group
            if self.layer_3_size > 0 and self.layer_2_size > 0 and self.layer_1_size > 0:
                model.add(Dense(self.layer_3_size, activation='relu'))
                if dropout > 0:
                    model.add(Dropout(self.dropout))

            # classifier layer
            model.add(Dense(self.data.num_classes, activation='softmax'))
            

        elif architecture == "image_MLP_trainable":
            
            #######################
            ### image_MLP_trainable
            #######################
            # image classification (single frame)
            # fine-tune pretrained CNN with MLP on top
            #
            # start off freezing base CNN layers then will unfreeze 
            # after each training round
            #
            # we will ultimately want to compare our best fine-tuned 
            # CNN as a feature extractor vs fixed ImageNet pretrained CNN features
            
            # check inputs
            assert self.sequence_length == 1, "image_MLP_trainable requires sequence length of 1"
            assert self.pretrained_model_name is not None, "image_MLP_trainable requires a pretrained_model_name input" 
            assert self.pooling is not None, "image_MLP_trainable requires a pooling input" 
            
            ### create data object for this architecture
            if self.verbose:
                self.logger.info("Loading data")
            self.data = Data(sequence_length = 1, 
                                return_CNN_features = False, 
                                pretrained_model_name = self.pretrained_model_name,
                                pooling = self.pooling,
                                return_generator = True,
                                batch_size = self.batch_size)
            
            # create the base pre-trained model
            model_base = load_pretrained_model(self.pretrained_model_name, pooling=self.pooling)
            

            # freeze base model layers (will unfreeze after train top)
            for l in model_base.layers:
                l.trainable=False

            # use Keras functional API
            model_top = model_base.output

            # note layer names are there so we can exclude those layers 
            # when setting base model layers to trainable

            # 1st layer group
            if self.layer_1_size > 0:
                model_top = Dense(self.layer_1_size, activation="relu", name='top_a')(model_top)
                if self.dropout > 0:
                    model_top = Dropout(self.dropout, name='top_b')(model_top)

            # 2nd layer group
            if self.layer_2_size > 0 and self.layer_1_size > 0:
                model_top = Dense(self.layer_2_size, activation="relu", name='top_c')(model_top)
                if self.dropout > 0:
                    model_top = Dropout(self.dropout, name='top_d')(model_top)

            # 3rd layer group
            if self.layer_3_size > 0 and self.layer_2_size > 0 and self.layer_1_size > 0:
                model_top = Dense(self.layer_3_size, activation="relu", name='top_e')(model_top)
                if self.dropout > 0:
                    model_top = Dropout(self.dropout, name='top_f')(model_top)

            # classifier layer
            model_predictions = Dense(self.data.num_classes, activation="softmax", name='top_g')(model_top)

            # combine base and top models into single model object
            model = Model(inputs=model_base.input, outputs=model_predictions)
                
        elif architecture == "video_MLP_concat":

            ####################
            ### video_MLP_concat
            ####################
            
            # concatenate all frames in sequence and train MLP on top of concatenated frame input
            
            assert self.sequence_length > 1, "video_MLP_concat requires sequence length > 1"
            assert self.pretrained_model_name is not None, "video_MLP_concat requires a pretrained_model_name input"
            assert self.pooling is not None, "video_MLP_concat requires a pooling input"
            
            ### create data object for this architecture
            if self.verbose:
                self.logger.info("Loading data")
            self.data = Data(sequence_length = self.sequence_length, 
                                return_CNN_features = True, 
                                pretrained_model_name=self.pretrained_model_name,
                                pooling = self.pooling)

            # init model
            model = Sequential()

            model.add(Flatten(input_shape=(self.sequence_length, self.num_features)))

            # 1st layer group
            if self.layer_1_size > 0:
                model.add(Dense(self.layer_1_size, activation='relu', input_shape=(self.num_features,)))
                if self.dropout > 0:
                    model.add(Dropout(self.dropout))

            # 2nd layer group
            if self.layer_2_size > 0 and self.layer_1_size > 0:
                model.add(Dense(self.layer_2_size, activation='relu'))
                if self.dropout > 0:
                    model.add(Dropout(self.dropout))

            # 3rd layer group
            if self.layer_3_size > 0 and self.layer_2_size > 0 and self.layer_1_size > 0:
                model.add(Dense(self.layer_3_size, activation='relu'))
                if self.dropout > 0:
                    model.add(Dropout(self.dropout))

            # classifier layer
            model.add(Dense(self.data.num_classes, activation='softmax'))
            
        elif architecture == "video_LRCNN_frozen":

            ######################
            ### video_LRCNN_frozen
            ######################
            
            # Implement:
            # “Long-Term Recurrent Convolutional Networks for Visual Recognition and Description.”
            # Donahue, Jeff, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan, 
            # Sergio Guadarrama, Kate Saenko, and Trevor Darrell.  
            # Proceedings of the IEEE Computer Society Conference on Computer Vision and 
            # Pattern Recognition, 2015, 2625–34.
            #
            # Essentially they extract features with fine-tuned CNN then fit recurrent models on top
            # in the paper they only use LSTM but we will also try RNN, GRU and 1-D CNN
            # 
            # note: no fine-tuning of CNN in this frozen LRCNN architecture
            # 
            # implementation inspired by:
            # https://github.com/sagarvegad/Video-Classification-CNN-and-LSTM-/blob/master/train_CNN_RNN.py

            
            # check inputs
            assert self.sequence_length > 1, "video_LRCNN_frozen requires sequence length > 1"
            assert self.layer_1_size > 0, "video_LRCNN_frozen requires a layer_1_size > 0" 
            assert self.pretrained_model_name is not None, "video_LRCNN_frozen requires a pretrained_model_name input" 
            assert self.pooling is not None, "video_LRCNN_frozen requires a pooling input" 
            assert self.sequence_model_layers is not None, "video_LRCNN_frozen requires sequence_model_layers >= 1" 
            assert self.sequence_model_layers >= 1, "video_LRCNN_frozen requires sequence_model_layers >= 1" 
            assert self.sequence_model_layers < 4, "video_LRCNN_frozen requires sequence_model_layers <= 3" 
            assert self.sequence_model is not None, "video_LRCNN_frozen requires a sequence_model" 
            if self.sequence_model == 'Convolution1D':
                assert self.convolution_kernel_size > 0, "Convolution1D sequence model requires convolution_kernel_size parameter > 0"
                assert self.convolution_kernel_size < self.sequence_length, "convolution_kernel_size must be less than sequence_length"

                
            ### create data object for this architecture
            if self.verbose:
                self.logger.info("Loading data")
            self.data = Data(sequence_length = self.sequence_length, 
                                return_CNN_features = True, 
                                pretrained_model_name = self.pretrained_model_name,
                                pooling = self.pooling)
            
                
            # set whether to return sequences for stacked sequence models
            return_sequences_1, return_sequences_2 = False, False
            if sequence_model_layers > 1 and layer_2_size > 0:
                return_sequences_1 = True
            if sequence_model_layers >= 2 and layer_3_size > 0 and layer_2_size > 0:
                return_sequences_2 = True
            
            # init model
            model = Sequential()

            # layer 1 (sequence layer)
            if sequence_model == "LSTM":
                model.add(LSTM(self.layer_1_size, return_sequences=return_sequences_1, dropout=self.dropout, 
                     
                               archinput_shape=(self.sequence_length, self.num_features)))
            elif sequence_model == "SimpleRNN":
                model.add(SimpleRNN(self.layer_1_size, return_sequences=return_sequences_1, dropout=self.dropout, 
                               input_shape=(self.sequence_length, self.num_features)))
            elif sequence_model == "GRU":
                model.add(GRU(self.layer_1_size, return_sequences=return_sequences_1, dropout=self.dropout, 
                               input_shape=(self.sequence_length, self.num_features)))
            elif sequence_model == "Convolution1D":
                model.add(Convolution1D(self.layer_1_size, kernel_size = self.convolution_kernel_size, padding = 'valid', 
                               input_shape=(self.sequence_length, self.num_features)))
                if layer_2_size == 0 or sequence_model_layers == 1:
                    model.add(Flatten())
            else:
                raise NameError('Invalid sequence_model - must be one of [LSTM, SimpleRNN, GRU, Convolution1D]')    

            # layer 2 (sequential or dense)
            if layer_2_size > 0:
                if return_sequences_1 == False:
                    model.add(Dense(self.layer_2_size, activation='relu'))
                    model.add(Dropout(self.dropout))
                else:
                    if sequence_model == "LSTM":
                        model.add(LSTM(self.layer_2_size, return_sequences=return_sequences_2, dropout=self.dropout))
                    elif sequence_model == "SimpleRNN":
                        model.add(SimpleRNN(self.layer_2_size, return_sequences=return_sequences_2, dropout=self.dropout))
                    elif sequence_model == "GRU":
                        model.add(GRU(self.layer_2_size, return_sequences=return_sequences_2, dropout=self.dropout))
                    elif sequence_model == "Convolution1D":
                        model.add(Convolution1D(self.layer_2_size, kernel_size = self.convolution_kernel_size, padding = 'valid'))
                    else:
                        raise NameError('Invalid sequence_model - must be one of [LSTM, SimpleRNN, GRU, Convolution1D]') 

            # layer 3 (sequential or dense)
            if layer_3_size > 0:
                if sequence_model_layers < 3:
                    if sequence_model_layers == 2:
                        model.add(Flatten())
                    model.add(Dense(self.layer_3_size, activation='relu'))
                    model.add(Dropout(self.dropout))
                else:
                    if sequence_model == "LSTM":
                        model.add(LSTM(self.layer_3_size, return_sequences=False, dropout=self.dropout))
                        model.add(Flatten())
                    elif sequence_model == "SimpleRNN":
                        model.add(SimpleRNN(self.layer_3_size, return_sequences=False, dropout=self.dropout))
                        model.add(Flatten())
                    elif sequence_model == "GRU":
                        model.add(GRU(self.layer_3_size, return_sequences=False, dropout=self.dropout))
                        model.add(Flatten())
                    elif sequence_model == "Convolution1D":
                        model.add(Convolution1D(self.layer_3_size, kernel_size = self.convolution_kernel_size, padding = 'valid'))
                        model.add(Flatten())
                    else:
                        raise NameError('Invalid sequence_model - must be one of [LSTM, SimpleRNN, GRU, Convolution1D]') 
            else:
                if return_sequences_2 == True: 
                    model.add(Flatten())

            # final flatten if needed
            if model.layers[-1].output_shape != (None, self.data.num_classes):
                model.add(Flatten())

            # classifier layer
            if self.dropout > 0:
                model.add(Dropout(self.dropout))
            model.add(Dense(self.data.num_classes, activation='softmax'))

        elif architecture == "video_LRCNN_trainable":
            
            #########################
            ### video_LRCNN_trainable
            #########################
            
            # Same as above:
            # “Long-Term Recurrent Convolutional Networks for Visual Recognition and Description.”
            # Donahue, Jeff, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan, 
            # Sergio Guadarrama, Kate Saenko, and Trevor Darrell.  
            # Proceedings of the IEEE Computer Society Conference on Computer Vision and 
            # Pattern Recognition, 2015, 2625–34.
            #
            # But with fine-tuning of the CNNs that are input into the recurrent models
            # 
            # note: will take long because not precomputing the CNN part so re-computed 
            # on each training pass

            # implementation inspired by https://stackoverflow.com/questions/49535488/lstm-on-top-of-a-pre-trained-cnn
            
            # check inputs
            assert self.sequence_length > 1, "video_LRCNN_trainable requires sequence length > 1"
            assert self.layer_1_size > 0, "video_LRCNN_trainable requires a layer_1_size > 0" 
            assert self.pretrained_model_name is not None, "video_LRCNN_trainable requires a pretrained_model_name input" 
            assert self.pooling is not None, "video_LRCNN_trainable requires a pooling input" 
            assert self.sequence_model_layers >= 1, "video_LRCNN_trainable requires sequence_model_layers >= 1" 
            assert self.sequence_model_layers < 4, "video_LRCNN_trainable requires sequence_model_layers <= 3" 
            assert self.sequence_model is not None, "video_LRCNN_trainable requires a sequence_model" 
            if self.sequence_model == 'Convolution1D':
                assert self.convolution_kernel_size > 0, "Convolution1D sequence model requires convolution_kernel_size parameter > 0"
                assert self.convolution_kernel_size < self.sequence_length, "convolution_kernel_size must be less than sequence_length"
                
                
            ### create data object for this architecture
            if self.verbose:
                self.logger.info("Loading data")
            self.data = Data(sequence_length = self.sequence_length, 
                                return_CNN_features = False, 
                                return_generator=True,
                                pretrained_model_name = self.pretrained_model_name,
                                pooling = self.pooling,
                                batch_size=self.batch_size)
            
                
            # set whether to return sequences for stacked sequence models
            return_sequences_1, return_sequences_2 = False, False
            if sequence_model_layers > 1 and layer_2_size > 0:
                return_sequences_1 = True
            if sequence_model_layers >= 2 and layer_3_size > 0 and layer_2_size > 0:
                return_sequences_2 = True

            # load pretrained model weights - will train from there
            model_cnn = load_pretrained_model(self.pretrained_model_name, pooling=self.pooling)

            # optionally load weights for pretrained architecture
            # (will likely be better to first train CNN then load weights in LRCNN vs. use pretrained ImageNet CNN)
            if self.model_weights_path is not None:
                model_base.load_weights(self.model_weights_path)
            
            # freeze model_cnn layers but make final 3 layers of pretrained CNN trainable
            for i, l in enumerate(model_cnn.layers):
                if i < len(model_cnn.layers)-3:
                    l.trainable = False
                else:
                    l.trainable = True

            # sequential component on top of CNN
            frames = Input(shape=(self.sequence_length, self.frame_size[0], self.frame_size[1], 3))
            x = TimeDistributed(model_cnn)(frames)
            x = TimeDistributed(Flatten())(x)
            

            # layer 1 (sequence layer)
            if sequence_model == "LSTM":
                x = LSTM(self.layer_1_size, return_sequences=return_sequences_1, dropout=self.dropout)(x)
            elif sequence_model == "SimpleRNN":
                x = SimpleRNN(self.layer_1_size, return_sequences=return_sequences_1, dropout=self.dropout)(x)
            elif sequence_model == "GRU":
                x = GRU(self.layer_1_size, return_sequences=return_sequences_1, dropout=self.dropout)(x)
            elif sequence_model == "Convolution1D":
                x = Convolution1D(self.layer_1_size, kernel_size = self.convolution_kernel_size, padding = 'valid')(x)
                if layer_2_size == 0 or sequence_model_layers == 1:
                    x = Flatten()(x)
            else:
                raise NameError('Invalid sequence_model - must be one of [LSTM, SimpleRNN, GRU, Convolution1D]')    

            # layer 2 (sequential or dense)
            if layer_2_size > 0:
                if return_sequences_1 == False:
                    x = Dense(self.layer_2_size, activation='relu')(x)
                    x = Dropout(self.dropout)(x)
                else:
                    if sequence_model == "LSTM":
                        x = LSTM(self.layer_2_size, return_sequences=return_sequences_2, dropout=self.dropout)(x)
                    elif sequence_model == "SimpleRNN":
                        x = SimpleRNN(self.layer_2_size, return_sequences=return_sequences_2, dropout=self.dropout)(x)
                    elif sequence_model == "GRU":
                        x = GRU(self.layer_2_size, return_sequences=return_sequences_2, dropout=self.dropout)(x)
                    elif sequence_model == "Convolution1D":
                        x = Convolution1D(self.layer_2_size, kernel_size = self.convolution_kernel_size, padding = 'valid')(x)
                    else:
                        raise NameError('Invalid sequence_model - must be one of [LSTM, SimpleRNN, GRU, Convolution1D]') 

            # layer 3 (sequential or dense)
            if layer_3_size > 0:
                if sequence_model_layers < 3:
                    if sequence_model_layers == 2:
                        x = Flatten()(x)
                    x = Dense(self.layer_3_size, activation='relu')(x)
                    x = Dropout(self.dropout)(x)
                else:
                    if sequence_model == "LSTM":
                        x = LSTM(self.layer_3_size, return_sequences=False, dropout=self.dropout)(x)
                        x = Flatten()(x)
                    elif sequence_model == "SimpleRNN":
                        x = SimpleRNN(self.layer_3_size, return_sequences=False, dropout=self.dropout)(x)
                        x = Flatten()(x)
                    elif sequence_model == "GRU":
                        x = GRU(self.layer_3_size, return_sequences=False, dropout=self.dropout)(x)
                        x = Flatten()(x)
                    elif sequence_model == "Convolution1D":
                        x = Convolution1D(self.layer_3_size, kernel_size = self.convolution_kernel_size, padding = 'valid')(x)
                        x = Flatten()(x)
                    else:
                        raise NameError('Invalid sequence_model - must be one of [LSTM, SimpleRNN, GRU, Convolution1D]') 
            else:
                if return_sequences_2 == True: 
                    x = Flatten()(x)

            # classifier layer
            if self.dropout > 0:
                x = Dropout(self.dropout)(x)
            out = Dense(self.data.num_classes, activation='softmax')(x)
                        

            # join cnn frame model and LSTM top
            model = Model(inputs=frames, outputs=out)
         
        elif architecture == "C3D":
            
            #########
            ### C3D
            #########
            
            # Implement:
            # Learning Spatiotemporal Features with 3D Convolutional Networks
            # Tran et al 2015
            # https://arxiv.org/abs/1412.0767
            #
            # Implementation inspired by https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
            
            assert self.sequence_length == 16, "C3D requires sequence length 16"
            assert self.frame_size == (112,112), "C3D requires frame size 112x112"
            assert self.layer_1_size == 0, "C3D does not accept layer size inputs since it's a predefined architecture"
            assert self.layer_2_size == 0, "C3D does not accept layer size inputs since it's a predefined architecture"
            assert self.layer_3_size == 0, "C3D does not accept layer size inputs since it's a predefined architecture"
            assert self.dropout == 0, "C3D does not accept layer size inputs since it's a predefined architecture"
            assert self.sequence_model == None, "C3D does not accept a sequence_model parameter"
            assert self.sequence_model_layers == None, "C3D does not accept a sequence_model_layers parameter"
            assert self.pretrained_model_name == None, "C3D does not accept a pretrained_model_name parameter"            
            assert self.pooling == None, "C3D does not accept a pooling parameter"                            
            
            
            ### create data object for this architecture
            if self.verbose:
                self.logger.info("Loading data")
            self.data = Data(sequence_length = 16, 
                                return_CNN_features = False, 
                                return_generator = True,
                                frame_size = (112,112),
                                batch_size=self.batch_size,
                                verbose = False)
            
            # C3D
            model = Sequential()
            # 1st layer group
            model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv1', input_shape=(16, 112, 112, 3)))
            model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))
            # 2nd layer group
            model.add(Conv3D(128, (3, 3, 3), activation='relu',padding='same', name='conv2'))
            model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),padding='valid', name='pool2'))
            # 3rd layer group
            model.add(Conv3D(256, (3, 3, 3), activation='relu',padding='same', name='conv3a'))
            model.add(Conv3D(256, (3, 3, 3), activation='relu',padding='same', name='conv3b'))
            model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))
            # 4th layer group
            model.add(Conv3D(512, (3, 3, 3), activation='relu',padding='same', name='conv4a'))
            model.add(Conv3D(512, (3, 3, 3), activation='relu',padding='same', name='conv4b'))
            model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),padding='valid', name='pool4'))
            # 5th layer group
            model.add(Conv3D(512, (3, 3, 3), activation='relu',padding='same', name='conv5a'))
            model.add(Conv3D(512, (3, 3, 3), activation='relu',padding='same', name='conv5b'))
            model.add(ZeroPadding3D(padding=(0, 1, 1)))
            model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),padding='valid', name='pool5'))
            model.add(Flatten())
            # FC layers group
            model.add(Dense(4096, activation='relu', name='fc6'))
            model.add(Dropout(.5))
            model.add(Dense(4096, activation='relu', name='fc7'))
            model.add(Dropout(.5))
            model.add(Dense(self.data.num_classes, activation='softmax', name='fc8'))
            
        elif architecture == "C3Dsmall":
            
            #########################
            ### C3D - small variation
            #########################
            
            # Custom small version of C3D from paper:
            # Learning Spatiotemporal Features with 3D Convolutional Networks
            # Tran et al 2015
            # https://arxiv.org/abs/1412.0767
            #
            # Implementation inspired by https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
            
            assert self.sequence_length == 16, "C3Dsmall requires sequence length 16"
            assert self.frame_size == (112,112), "C3Dsmall requires frame size 112x112"
            assert self.layer_1_size == 0, "C3Dsmall does not accept layer size inputs since it's a predefined architecture"
            assert self.layer_2_size == 0, "C3Dsmall does not accept layer size inputs since it's a predefined architecture"
            assert self.layer_3_size == 0, "C3Dsmall does not accept layer size inputs since it's a predefined architecture"
            assert self.dropout == 0, "C3Dsmall does not accept layer size inputs since it's a predefined architecture"
            assert self.sequence_model == None, "C3Dsmall does not accept a sequence_model parameter"
            assert self.sequence_model_layers == None, "C3Dsmall does not accept a sequence_model_layers parameter"
            assert self.pretrained_model_name == None, "C3Dsmall does not accept a pretrained_model_name parameter"            
            assert self.pooling == None, "C3Dsmall does not accept a pooling parameter"      
            
            
            ### create data object for this architecture
            if self.verbose:
                self.loggerinfo("Loading data")
            self.data = Data(sequence_length = 16, 
                                return_CNN_features = False, 
                                return_generator = True,
                                frame_size = (112,112),
                                batch_size=self.batch_size,
                                verbose = False)
            
            # C3Dsmall
            model = Sequential()
            # 1st layer group
            model.add(Conv3D(32, (3,3,3), activation='relu', input_shape=(data.sequence_length, 112, 112, 3)))
            model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
            # 2nd layer group
            model.add(Conv3D(64, (3,3,3), activation='relu'))
            model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
            # 3rd layer group
            model.add(Conv3D(128, (3,3,3), activation='relu'))
            model.add(Conv3D(128, (3,3,3), activation='relu'))
            model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
            # 4th layer group
            model.add(Conv3D(256, (2,2,2), activation='relu'))
            model.add(Conv3D(256, (2,2,2), activation='relu'))
            model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
            # FC layers group
            model.add(Flatten())
            model.add(Dense(256))
            model.add(Dropout(0.5))
            model.add(Dense(128))
            model.add(Dropout(0.5))
            model.add(Dense(self.data.num_classes, activation='softmax'))
            
        else:
            raise NameError('Invalid architecture - must be one of [image_MLP_frozen, image_MLP_trainable, video_MLP_concat, video_LRCNN_frozen, video_LRCNN_trainable, C3D, C3Dsmall]')    
        
        
        
        
        ###############
        ### Finish init
        ###############
        
        # set class model to constructed model
        self.model = model
        
        # load weights of model if they exist
        if os.path.exists(self.path_model + 'model.h5'):
            if self.verbose:
                self.loggerinfo("Loading saved model weights")
            # model.load_weights(self.path_model + 'model.h5')            
            model = load_model(self.path_model + 'model.h5')
        
        # save architecture params to model folder
        params = self.__dict__.copy()
        params['data_shape'] = str(self.data)
        # remove non-serializable objects first
        del params['model']
        del params['data']
        del params['logger']
        # save
        with open(self.path_model + 'params.json', 'w') as fp:
            json.dump(params, fp, indent=4, sort_keys=True)
    
    
    def make_last_layers_trainable(self, num_layers):
        """
        Set the last *num_layers* non-trainable layers to trainable  

        NB to be used with model_base and assumes name = "top_xxx" added to each top layer to know 
        to ignore that layer when looping through layers from top backwards

        :num_layers: number of layers from end of model (that are currently not trainable) to be set as trainable
        """

        # get index of last non-trainable layer
        # (the layers we added on top of model_base are already trainable=True)
        # ...
        # need to find last layer of base model and set that (and previous num_layers)
        # to trainable=True via this method

        # find last non-trainable layer index
        idx_not_trainable = 0
        for i, l in enumerate(self.model.layers):
            if "top" not in l.name:
                idx_not_trainable = i

        # set last non-trainable layer and num_layers prior to trainable=True
        for i in reversed(range(idx_not_trainable-num_layers+1, idx_not_trainable+1)):
            self.model.layers[i].trainable = True
        
        if self.verbose:
            self.loggerinfo("last {} layers of CNN set to trainable".format(num_layers))
            

    def fit(self, fit_round, learning_rate, epochs, patience):
        """
        Compile and fit model for *epochs* rounds of training, dividing learning rate by 10 after each round

        Fitting will stop if val_acc does not improve for at least patience epochs

        Only the best weights will be kept

        The model is saved to /models/*model_id*/

        Good practice is to decrease the learning rate by a factor of 10 after each plateau and train some more 
        (after first re-loading best weights from previous training round)...

        for example (not exact example because this fit method has been refactored into the architecture object but the principle remains):
            fit_history = fit(model_id, model, data, learning_rate = 0.001, epochs = 30)
            model.load_weights(path_model + "model.h5")
            model = fit(model, 5)
            fit_history = train(model_id, model, data, learning_rate = 0.0001, epochs = 30)

        :fit_round: keep track of which round of learning rate annealing we're on
        :learning_rate: learning rate parameter for Adam optimizer (default is 0.001)
        :epochs: number of training epochs per fit round, subject to patience setting - good default is 30 or more
        :patience: how many epochs without val_acc improvement before stopping fit round (good default is 5) 
        
        :verbose: print progress

        """

        # create optimizer with given learning rate 
        opt = Adam(lr = learning_rate)

        # compile model
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # setup training callbacks
        callback_stopper = EarlyStopping(monitor='val_acc', patience=patience, verbose=self.verbose)
        callback_csvlogger = CSVLogger(self.path_model + 'training_round_' + str(fit_round) + '.log')
        callback_checkpointer = ModelCheckpoint(self.path_model + 'model_round_' + str(fit_round) + '.h5', monitor='val_acc', save_best_only=True, verbose=self.verbose)
        callbacks = [callback_stopper, callback_checkpointer, callback_csvlogger]

        # fit model
        if self.data.return_generator == True:
            # train using generator
            history = self.model.fit_generator(generator=self.data.generator_train,
                validation_data=self.data.generator_valid,
                use_multiprocessing=True,
                workers=CPU_COUNT,
                epochs=epochs,
                callbacks=callbacks,
                shuffle=True,
                verbose=self.verbose)
        else:
            # train using full dataset
            history = self.model.fit(self.data.x_train, self.data.y_train, 
                validation_data=(self.data.x_valid, self.data.y_valid),
                batch_size=self.batch_size,
                epochs=epochs,
                callbacks=callbacks,
                shuffle=True,
                verbose=self.verbose)

        # get number of epochs actually trained (might have early stopped)
        epochs_trained = callback_stopper.stopped_epoch
        
        if epochs_trained == 0:
            # trained but didn't stop early
            if len(history.history) > 0:
                epochs_trained = (epochs - 1)
        else:
            # best validation accuracy is (patience-1) epochs before stopped
            epochs_trained -= (patience - 1)
            
           
        
        # return fit history and the epoch that the early stopper completed on
        return history, epochs_trained

    
    def train_model(self, epochs = 20, patience = 3):
        """
        Run several rounds of fitting to train model, reducing learning rate after each round
        
        Progress and model parameters will be saved to the model's path e.g. /models/1/
        
        """
        
        # init results with architecture params
        results = self.__dict__.copy()
        results['data_total_rows_train'] = self.data.total_rows_train
        results['data_total_rows_valid'] = self.data.total_rows_valid
        results['data_total_rows_test'] = self.data.total_rows_test
        # delete non-serializable objects from the architecture class
        del results['model']
        del results['data']
        del results['logger']
        results['model_param_count'] = self.model.count_params()
        
        
        ###############
        ### Train model
        ###############
        
        # start training timer
        start = datetime.datetime.now()
        results['fit_dt_train_start'] = start.strftime("%Y-%m-%d %H:%M:%S")
        
        
        ### Fit round 1
        
        # do first round of fitting
        history1, stopped_epoch1 = self.fit(fit_round = 1, learning_rate = 0.001, epochs = epochs, patience = patience)
        
        print('H1', history1.history)
        print('stopped_epoch1',stopped_epoch1)
        print(len(history1.history['val_acc']))
        print(history1.history['val_acc'][stopped_epoch1])
        
        # update best fit round (only 1 round done so this is best so far)
        best_val_acc_1 = history1.history['val_acc'][stopped_epoch1]
        best_fit_round = 1
        best_fit_round_val_acc = best_val_acc_1
        #
        best_fit_round_train_acc = history1.history['acc'][stopped_epoch1]
        best_fit_round_train_loss = history1.history['loss'][stopped_epoch1]
        best_fit_round_val_loss = history1.history['val_loss'][stopped_epoch1]
        
        
        ### Fit round 2
        
        # load best model weights so far
        model = load_model(self.path_model + 'model_round_' + str(best_fit_round) + '.h5')
        
        # reduce learning rate and fit some more
        history2, stopped_epoch2 = self.fit(fit_round = 2, learning_rate = 0.0001, epochs = epochs, patience = patience)
        
        print('H2', history2.history)
        print('stopped_epoch2',stopped_epoch2)
        print(len(history2.history['val_acc']))
        print(history2.history['val_acc'][stopped_epoch2])
        
        # update best fit round
        best_val_acc_2 = history2.history['val_acc'][stopped_epoch2]
        if best_val_acc_2 > best_fit_round_val_acc:
            best_fit_round_val_acc = best_val_acc_2
            best_fit_round_train_acc = history2.history['acc'][stopped_epoch2]
            best_fit_round_train_loss = history2.history['loss'][stopped_epoch2]
            best_fit_round_val_loss = history2.history['val_loss'][stopped_epoch2]
            best_fit_round = 2
            
            
        ### Fit round 3
        
        # load best model weights so far
        model = load_model(self.path_model + 'model_round_' + str(best_fit_round) + '.h5')
        
        # reduce learning rate and fit some more
        history3, stopped_epoch3 = self.fit(fit_round = 3, learning_rate = 0.00001, epochs = epochs, patience = patience)
        
        print('H3', history3.history)
        print('stopped_epoch3',stopped_epoch3)
        print(len(history3.history['val_acc']))
        print(history3.history['val_acc'][stopped_epoch3])
        
        # update best fit round
        best_val_acc_3 = history3.history['val_acc'][stopped_epoch3]
        if best_val_acc_3 > best_fit_round_val_acc:
            best_fit_round_val_acc = best_val_acc_3
            best_fit_round_train_acc = history3.history['acc'][stopped_epoch3]
            best_fit_round_train_loss = history3.history['loss'][stopped_epoch3]
            best_fit_round_val_loss = history3.history['val_loss'][stopped_epoch3]
            best_fit_round = 3
        
        
        ### Finish fit process
        
        # end time training
        end = datetime.datetime.now()    
        results['fit_dt_train_end']  = end.strftime("%Y-%m-%d %H:%M:%S")
        results['fit_dt_train_duration_seconds']  = str((end - start).total_seconds()).split(".")[0]
        
        # set best weights file across the fit rounds
        print("best fit round",best_fit_round, best_fit_round_val_acc)
        copyfile(self.path_model + 'model_round_' + str(best_fit_round) + '.h5', self.path_model + 'model_best.h5')
        
        #################
        ### build results
        #################
        # combine fit histories into big dataframe and write to model folder
        # only keep history until accuracy declined (where early stopping made checkpoint)

        # parse history dicts to dataframes
        history1 = pd.DataFrame(history1.history).head(stopped_epoch1)
        history1['fit_round'] = 1
        history2 = pd.DataFrame(history2.history).head(stopped_epoch2)
        history2['fit_round'] = 2
        history3 = pd.DataFrame(history3.history).head(stopped_epoch3)
        history3['fit_round'] = 3
        
        # combine and save csv
        fit_history = pd.concat([history1, history2, history3], axis=0)
        fit_history = fit_history.reset_index(drop=True)
        fit_history['epoch'] = fit_history.index+1
        fit_history.to_csv(self.path_model + 'fit_history.csv')
        self.fit_history = fit_history
        
        results['fit_stopped_epoch1'] = stopped_epoch1
        results['fit_stopped_epoch2'] = stopped_epoch2
        results['fit_stopped_epoch3'] = stopped_epoch3
        
        # add 3 = 1 for each training round because stopped_epoch is 0 indexed
        results['fit_num_epochs'] = stopped_epoch1 + stopped_epoch2 + stopped_epoch3 + 3
        results['fit_val_acc'] = best_fit_round_val_acc
        results['fit_train_acc'] = best_fit_round_train_acc
        results['fit_val_loss'] = best_fit_round_val_loss
        results['fit_train_loss'] = best_fit_round_train_loss
        results['fit_best_round'] = best_fit_round
        
        # save model summary to model folder
        with open(self.path_model + 'model_summary.txt', 'w') as f:
            self.model.summary()
                
        # save model config to model folder
        self.model.save(self.path_model + "model_config.h5")

        #######################
        ### Predict on test set
        #######################
        
        # start test timer
        start = datetime.datetime.now()
        results['fit_dt_test_start'] = start.strftime("%Y-%m-%d %H:%M:%S")
        
        y_pred = None
        y_test = None
        if self.data.return_generator:
            # predict on test set via generator
            y_pred = self.model.predict_generator(self.data.generator_test,verbose=self.verbose)
            
            # save predicted clas probabilities
            np.save(self.path_model + 'test_predictions', y_pred)
            
            # take argmax to get predicted class
            y_pred = np.argmax(y_pred, axis = 1)

            # get truth labels from generator
            y_test = []
            for _, label in self.data.generator_test:
                y_test.extend(label)
            y_test = np.argmax(np.array(y_test), axis = 1)
            
        else:
            # predict on test data loaded into memory
            y_pred = self.model.predict(self.data.x_test, verbose=self.verbose)
            
            # save predicted clas probabilities
            np.save(self.path_model + 'test_predictions', y_pred)
            
            # take argmax to get predicted class
            y_pred = np.argmax(y_pred, axis=1)

            # get truth labels from memory
            y_test = np.argmax(self.data.y_test,axis=1)
        
        # end time testing
        end = datetime.datetime.now()    
        results['fit_dt_test_end']  = end.strftime("%Y-%m-%d %H:%M:%S")
        results['fit_dt_test_duration_seconds']  = str((end - start).total_seconds()).split(".")[0]
        
        ############################
        ### Compute confusion matrix
        ############################
        
        # Compute and store confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        pd.DataFrame(cnf_matrix).to_csv(self.path_model + "confusion_matrix.csv")

        # get clas names from label map for plot
        class_names = list(self.data.label_map.values())

        # Plot non-normalized confusion matrix
        plt.figure(figsize=(8,8))
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
        plt.savefig(self.path_model + 'confusion_matrix.png', bbox_inches='tight')
        plt.clf()

        # Plot normalized confusion matrix
        plt.figure(figsize=(8,8))
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
        plt.savefig(self.path_model + 'confusion_matrix_normalized.png', bbox_inches='tight')
        plt.clf()
        
        ##########################
        ### Compute raw error rate
        ##########################
        
        # build dataframe and calculate test error (assuming we classify using majority rule, not ROC cutoff approach)
        pdf = pd.DataFrame(y_pred, columns = ['pred'])
        pdf['prediction'] = pdf['pred'].apply(lambda x: self.data.label_map[str(x)])

        truth = pd.DataFrame(y_test, columns = ['truth'])
        truth['label'] = truth['truth'].apply(lambda x: self.data.label_map[str(x)])
        truth = truth[['label']]

        pdf = pd.concat([pdf, truth], axis=1)
        pdf['error'] = (pdf['prediction'] != pdf['label']).astype(int)
        test_acc = 1 - pdf['error'].mean()
        
        results['fit_test_acc'] = test_acc
        
        if self.verbose:
            self.logger.info(json.dumps(results, indent=4, sort_keys=True))
            self.logger.info("model {} test acc: {}".format(self.model_id, test_acc))
        
        
        ##################
        ### Output results
        ##################
        self.results = results
        with open(self.path_model + 'results.json', 'w') as fp:
            json.dump(results, fp, indent=4, sort_keys=True)
            
        # sync model outputs to s3
        response = os.system("aws s3 sync " + self.path_model + " s3://thesisvids/penguins/models/" + str(self.model_id) + "/")
        if response != 0:
            logging.error("ERROR syncing model_id = {}".format(self.model_id))

        