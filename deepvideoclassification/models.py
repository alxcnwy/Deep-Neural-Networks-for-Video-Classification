#!/usr/bin/env python
# coding: utf-8

# In[2]:


### TODO
# * 3-d CNN
# * concat dense model
# * fit_models create_architectures_list (append mode)
# * fit_models worker if experiment id last digit in os environment var
# * run penguin preprocessing



# * refactor custom_model_name and model_weights_path to instead use trained model id


# In[3]:


# whether to log each feature and sequence status
verbose = 1


# In[4]:


import os
import pandas as pd
import numpy as np
from PIL import Image
import json
import cv2
import sys
sys.path.append('..')


# In[7]:


from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, Convolution1D, Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import img_to_array


# In[8]:


# setup paths
pwd = os.getcwd().replace("deepvideoclassification","")
path_cache = pwd + 'cache/'
path_data = pwd + 'data/'


# In[9]:


# setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(pwd, "logs")),
        logging.StreamHandler()
    ])
logger = logging.getLogger()


# # Pretrained_CNNs

# In[ ]:


# pretrained model shapes
pretrained_model_len_features = {}
#
pretrained_model_len_features['vgg16'] = 512
#
pretrained_model_len_features['mobilenetv2_1.00_224'] = 1280
#
pretrained_model_len_features['inception_resnet_v2'] = 1536
#
pretrained_model_len_features['resnet50'] = 2048
pretrained_model_len_features['xception'] = 2048
pretrained_model_len_features['inception_v3'] = 2048


# In[ ]:


# pretrained model shapes
pretrained_model_sizes = {}
#
pretrained_model_sizes['vgg16'] = (224,224)
pretrained_model_sizes['resnet50'] = (224,224)
pretrained_model_sizes['mobilenetv2_1.00_224'] = (224,224)
#
pretrained_model_sizes['xception'] = (299,299)
pretrained_model_sizes['inception_v3'] = (299,299)
pretrained_model_sizes['inception_resnet_v2'] = (299,299)


# In[ ]:


pretrained_model_names = ["inception_resnet_v2", "inception_v3", "mobilenetv2_1.00_224", "resnet50", "vgg16", "xception"]
poolings = ['max','avg']


# In[ ]:


def load_pretrained_model(pretrained_model_name, pooling, model_weights_path = None):
    """ Load pretrained model with given pooling applied
    
    Args:
        pretrained_model: name of pretrained model ["Xception", "VGG16", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNetV2"]
        pooling: pooling strategy for final pretrained model layer ["max","avg"]
        :model_weights_path: path to custom model weights if we want to load CNN model we've fine-tuned to produce features (e.g. for LRCNN)
    
    Returns:
        Pretrained model object (excluding dense softmax 1000 ImageNet classes layer)
    """
    
    # initialize output
    model = None
    
    pretrained_model_name = pretrained_model_name.lower()
    
    ###########################
    ### import pretrained model
    ###########################
    if pretrained_model_name == "xception":   
        from keras.applications.xception import Xception
        model = Xception(include_top=False, weights='imagenet', pooling=pooling)
    elif pretrained_model_name == "vgg16":   
        from keras.applications.vgg16 import VGG16
        model = VGG16(include_top=False, weights='imagenet', pooling=pooling)
    elif pretrained_model_name == "resnet50":   
        from keras.applications.resnet50 import ResNet50
        model = ResNet50(include_top=False, weights='imagenet', pooling=pooling)
    elif pretrained_model_name == "inception_v3":   
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(include_top=False, weights='imagenet', pooling=pooling)
    elif pretrained_model_name == "inception_resnet_v2":   
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        model = InceptionResNetV2(include_top=False, weights='imagenet', pooling=pooling)
    elif pretrained_model_name == "mobilenetv2_1.00_224":   
        from keras.applications.mobilenet_v2 import MobileNetV2
        model = MobileNetV2(include_top=False, weights='imagenet', pooling=pooling)
    else:
        raise NameError('Invalid pretrained model name - must be one of ["Xception", "VGG16", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNetV2"]')
    
    if model_weights_path is not None:
        if os.path.exists(model_weights_path):
            model.load_weights(model_weights_path)
        else:
            raise NameError('pretrained model weights not found')
    
    return model


# In[ ]:


def load_pretrained_model_preprocessor(pretrained_model_name):
    """
    Return preprocessing function for a given pretrained model
    """

    preprocess_input = None

    pretrained_model_name = pretrained_model_name.lower()
        
    if pretrained_model_name == "xception":   
        from keras.applications.xception import preprocess_input
    elif pretrained_model_name == "vgg16":   
        from keras.applications.vgg16 import preprocess_input
    elif pretrained_model_name == "resnet50":   
        from keras.applications.resnet50 import preprocess_input
    elif pretrained_model_name == "inception_v3":   
        from keras.applications.inception_v3 import preprocess_input
    elif pretrained_model_name == "inception_resnet_v2":   
        from keras.applications.inception_resnet_v2 import preprocess_input
    elif pretrained_model_name == "mobilenetv2_1.00_224":   
        from keras.applications.mobilenet_v2 import preprocess_input
    else:
        raise NameError('Invalid pretrained model name - must be one of ["Xception", "VGG16", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNetV2"]')
        
    return preprocess_input


# In[19]:


def precompute_CNN_features(pretrained_model_name, pooling, model_weights_path = None, custom_model_name = None):
    """ 
    Save pretrained features array computed over all frames of each video 
    using given pretrained model and pooling method
    
    :pretrained_model_name: pretrained model object loaded using `load_pretrained_model`
    :pooling: pooling method used with pretrained model
    :model_weights_path: path to custom model weights if we want to load CNN model we've fine-tuned to produce features (e.g. for LRCNN)
    :custom_model_name: custom output name to append to pretrained model name

    """
    
    pretrained_model_name = pretrained_model_name.lower()
    
    # setup path to save features
    path_features = path_cache + 'features/' + pretrained_model_name + "/" + pooling + '/'
    
    # store in custom directory if custom model name given (for when loading weights from fine-tuned CNN and precomputing features from that model)
    if custom_model_name is not None and model_weights_path is not None:
        path_features = path_cache + 'features/' + pretrained_model_name + "__" + custom_model_name + "/" + pooling + '/'
    
    if not os.path.exists(path_features):
        
        os.makedirs(path_features)
        
        # load pretrained model
        pretrained_model = load_pretrained_model(pretrained_model_name, pooling, model_weights_path)

        # load preprocessing function
        preprocess_input = load_pretrained_model_preprocessor(pretrained_model_name)

        # lookup pretrained model input shape
        model_input_size = pretrained_model_sizes[pretrained_model_name]
        
        # precompute features for each video using pretrained model
        from deepvideoclassification.data import get_video_paths
        path_videos = get_video_paths()

        for c, path_video in enumerate(path_videos):

            if verbose:
                logging.info("Computing pretrained model features for video {}/{} using pretrained model: {}, pooling: {}".format(c+1,len(path_videos),pretrained_model_name, pooling))

            # get video name from video path
            video_name = path_video.split("/")[-2]

            # build output path
            path_output = path_features + video_name

            try:
                if not os.path.exists(path_output + '.npy'):

                    path_frames = path_data + video_name + "/"

                    # initialize features list
                    features = []

                    frame_paths = os.listdir(path_frames)
                    frame_paths = [path_frames + f for f in frame_paths if f != '.DS_Store']

                    # sort paths in sequence (they were created with incrementing filenames through time)
                    frame_paths.sort()

                    # load each frame in vid and get features
                    for j, frame_path in enumerate(frame_paths):

                        # load image & preprocess
                        image = cv2.imread(frame_path, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(image, model_input_size, interpolation=cv2.INTER_AREA)
                        img = img_to_array(img)
                        img = np.expand_dims(img, axis=0)
                        img = preprocess_input(img)

                        # get features from pretrained model
                        feature = pretrained_model.predict(img).ravel()
                        features.append(feature)

                    # convert to arrays
                    features = np.array(features)

                    np.save(path_output, features)
                else:
                    if verbose:
                        logger.info("Features already cached: {}".format(path_output))

            except Exception as e:
                logging.error("Error precomputing features {} / {},{}".format(video_namepretrained_model_name, pooling))
                logging.fatal(e, exc_info=True)
                
    else:
        if verbose:
            logger.info("Features already cached: {}".format(path_features))


# # Image/video classification architecture object (contains keras model object) 

# In[ ]:


class Architecture(object):
    
    def __init__(self, architecture, sequence_length, num_classes, frame_size, 
                pretrained_model_name = None, pooling = None,
                sequence_model = None, sequence_model_layers = 1,
                layer_1_size = 0, layer_2_size = 0, layer_3_size = 0, 
                dropout = 0, convolution_kernel_size = 0, model_weights_path = None):
        """
        Model object constructor. Architecture can be one of: 
        image_MLP_frozen, image_MLP_trainable, video_MLP_concat, 
        video_LRCNN_frozen, video_LRCNN_trainable, 3DCNN
        
        :architecture: architecture of model in [image_MLP_frozen, image_MLP_trainable, video_MLP_concat, video_LRCNN_frozen, video_LRCNN_trainable, 3DCNN]
        
        :sequence_length: number of frames in sequence to be returned by Data object
        :num_classes: number of classes to predict
        :frame_size: size that frames are resized to (different models / architectures accept different input sizes)

        :pretrained_model_name: name of pretrained model (or None if not using pretrained model e.g. for 3D-CNN)
        :pooling: name of pooling variant (or None if not using pretrained model e.g. for 3D-CNN or if fitting more non-dense layers on top of pretrained model base)
        
        :sequence_model: sequence model in [LSTM, SimpleRNN, GRU, Convolution1D]
        :sequence_model_layers: default to 1, can be stacked 2 or 3 (but less than 4) layer sequence model (assume always stacking the same sequence model, not mixing LSTM and GRU, for example)
        
        :layer_1_size: number of neurons in layer 1
        :layer_2_size: number of neurons in layer 2
        :layer_3_size: number of neurons in layer 3 
        
        :dropout: amount of dropout to add (same applied throughout model - good default is 0.2) 
        
        :convolution_kernel_size: size of 1-D convolutional kernel for 1-d conv sequence models (good default is 3)
        
        :model_weights_path: path to .h5 weights file to be loaded for pretrained CNN in LRCNN-trainable and in 3d-CNN. Can use custom CNN for other models but need to save features first then load them in data
        """
    
        # required params
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.num_classes = num_classes
        
        # model architecture params
        self.architecture = architecture
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
        
        model = None
        
        if architecture == "image_MLP_frozen":
            
            ####################
            ### image_MLP_frozen
            ####################
            # image classification (single frame)
            # train MLP on top of weights extracted from pretrained CNN with no fine-tuning
            
            # check inputs
            assert self.sequence_length == 1, "image_MLP_frozen requires sequence length of 1"
            assert self.pretrained_model_name is not None, "image_MLP_frozen requires a pretrained_model_name input" 
            
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
            model.add(Dense(self.num_classes, activation='softmax'))

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
            
            
            # create the base pre-trained model
            model_base = load_pretrained_model(self.pretrained_model_name, pooling=self.pooling)

            # freeze base model layers (will unfreeze after train top)
            for l in model_base.layers:
                l.trainable=False

            # use Keras functional API
            model_top = base_model.output

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
            model_predictions = Dense(self.num_classes, activation="softmax", name='top_g')(model_top)

            # combine base and top models into single model object
            model = Model(inputs=base_model.input, outputs=model_predictions)
                
        elif architecture == "video_MLP_concat":

            ####################
            ### video_MLP_concat
            ####################
            
            # video classification
            # concatenate all frames in sequence and train MLP on top of concatenated frame input
            
            print('TODO')
            
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
            assert self.sequence_model_layers >= 1, "video_LRCNN_frozen requires sequence_model_layers >= 1" 
            assert self.sequence_model_layers < 4, "video_LRCNN_frozen requires sequence_model_layers <= 3" 
            assert self.sequence_model is not None, "video_LRCNN_frozen requires a sequence_model" 
            if self.sequence_model == 'Convolution1D':
                assert self.convolution_kernel_size > 0, "Convolution1D sequence model requires convolution_kernel_size parameter > 0"
                assert self.convolution_kernel_size < self.sequence_length, "convolution_kernel_size must be less than sequence_length"

            # set whether to return sequences for stacked sequence models
            return_sequences_1, return_sequences_2 = False, False
            if sequence_model_layers > 1 and layer_2_size > 0:
                return_sequences_1 = True
            if sequence_model_layers == 2 and layer_3_size > 0 and layer_2_size > 0:
                return_sequences_2 = True
                
            print(return_sequences_1, return_sequences_2)
                
            #LSTM, SimpleRNN, GRU, Convolution1D
            
            # init model
            model = Sequential()

            # layer 1 (sequence layer)
            if sequence_model == "LSTM":
                model.add(LSTM(self.layer_1_size, return_sequences=return_sequences_1, dropout=self.dropout, 
                               input_shape=(self.sequence_length, self.num_features)))
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
                    model.add(Dropout(self.dropout))
                    model.add(Dense(self.layer_2_size, activation='relu'))
                    model.add(Flatten())
                else:
                    if sequence_model == "LSTM":
                        model.add(LSTM(self.layer_2_size, return_sequences=return_sequences_2, dropout=self.dropout))
                    elif sequence_model == "SimpleRNN":
                        model.add(SimpleRNN(self.layer_2_size, return_sequences=return_sequences_2, dropout=self.dropout))
                    elif sequence_model == "GRU":
                        model.add(GRU(self.layer_2_size, return_sequences=return_sequences_2, dropout=self.dropout))
                    elif sequence_model == "Convolution1D":
                        model.add(Convolution1D(self.layer_2_size, kernel_size = self.convolution_kernel_size, padding = 'valid'))
                        if layer_3_size == 0 or sequence_model_layers == 2:
                            model.add(Flatten())
                    else:
                        raise NameError('Invalid sequence_model - must be one of [LSTM, SimpleRNN, GRU, Convolution1D]') 

            # layer 3 (sequential or dense)
            if layer_3_size > 0:
                if sequence_model_layers < 3:
                    model.add(Dropout(self.dropout))
                    model.add(Dense(self.layer_3_size, activation='relu'))
                    model.add(Flatten())
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

            # classifier layer
            if self.dropout > 0:
                model.add(Dropout(self.dropout))
            model.add(Dense(self.num_classes, activation='softmax'))

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

            model_cnn = load_pretrained_model(self.pretrained_model_name, pooling=self.pooling)

            # optionally load weights for pretrained architecture
            # (will likely be better to first train CNN then load weights in LRCNN vs. use pretrained ImageNet CNN)
            if self.model_weights_path is not None:
                base_model.load_weights(self.model_weights_path)
            
            # freeze model_cnn layers (will unfreeze later after sequence model trained a while)
            for l in model_cnn.layers:
                l.trainable = False

            # sequential component on top of CNN
            frames = Input(shape=(self.sequence_length, self.frame_size[0], self.frame_size[1], 3))
            x = TimeDistributed(model_cnn)(frames)
            x = TimeDistributed(Flatten())(x)

            # layer 1 sequence model
            x = LSTM(self.layer_1_size, dropout=dropout)(x)

            # classifier layer
            out = Dense(self.num_classes)(x)

            # join cnn frame model and LSTM top
            model = Model(inputs=frames, outputs=out)
            

            
        elif architecture == "3DCNN":
            
            #########
            ### 3DCNN
            #########
            
            # Implement:
            
            # “3D Convolutional Neural Networks for Human Action Recognition.” 
            # Ji, Shuiwang, Wei Xu, Ming Yang, and Kai Yu. 
            # IEEE Transactions on Pattern Analysis and Machine Intelligence 
            # 35, no. 1 (2013): 221–31. doi:10.1109/TPAMI.2012.59.
            #
            # They fit a 3-D convolutional model on top of stacked frame volumes
            
            # Implementation from: 
            # https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
            # note example has input shape as (channels, sequence_length, frame_size_1, frame_size_2)
            
            # init model
            model = Sequential()
            
            # 1st layer group
            model.add(Convolution3D(64, 3, 3, 3, activation='relu',  border_mode='same', name='conv1', subsample=(1, 1, 1), input_shape=(3, 16, 112, 112)))
            model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1'))

            # 2nd layer group
            model.add(Convolution3D(128, 3, 3, 3, activation='relu',border_mode='same', name='conv2', subsample=(1, 1, 1)))
            model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  border_mode='valid', name='pool2'))

            # 3rd layer group
            model.add(Convolution3D(256, 3, 3, 3, activation='relu',border_mode='same', name='conv3a', subsample=(1, 1, 1)))
            model.add(Convolution3D(256, 3, 3, 3, activation='relu',  border_mode='same', name='conv3b', subsample=(1, 1, 1)))
            model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3'))

            # 4th layer group
            model.add(Convolution3D(512, 3, 3, 3, activation='relu',  border_mode='same', name='conv4a', subsample=(1, 1, 1)))
            model.add(Convolution3D(512, 3, 3, 3, activation='relu',  border_mode='same', name='conv4b', subsample=(1, 1, 1)))
            model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4'))

            # 5th layer group
            model.add(Convolution3D(512, 3, 3, 3, activation='relu',border_mode='same', name='conv5a', subsample=(1, 1, 1)))
            model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5b', subsample=(1, 1, 1)))
            model.add(ZeroPadding3D(padding=(0, 1, 1)))
            model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool5'))
            model.add(Flatten())
            
            # FC layers group
            model.add(Dense(4096, activation='relu', name='fc6'))
            model.add(Dropout(.5))
            model.add(Dense(4096, activation='relu', name='fc7'))
            model.add(Dropout(.5))
            
            
            # if load weights from Sports1M model then need to load with 487 class classifier then pop it and add our own
            if self.model_weights_path is not None:
                model.add(Dense(487, activation='softmax', name='fc8'))
                model.load_weights(self.model_weights_path)
                model.layers.pop()
                model.add(Dense(self.num_classes, activation='softmax'))
            else:
                model.add(Dense(self.num_classes, activation='softmax', name='fc8'))
            
        else:
            raise NameError('Invalid architecture - must be one of [image_MLP_frozen, image_MLP_trainable, video_MLP_concat, video_LRCNN_frozen, video_LRCNN_trainable, 3DCNN]')    
        
        # set class model to constructed model
        self.model = model


# In[11]:


def make_last_layers_trainable(model, num_layers):
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
    idx_non_trainable = 0
    for i, l in enumerate(model.layers):
        if "top" not in l.name:
            idx_non_trainable = i
                
    # set last non-trainable layer and num_layers prior to trainable=True
    for i in reversed(range(idx_not_trainable-num_layers+1, idx_not_trainable+1)):
        model.layers[i].trainable = True
        print(idx_not_trainable, num_layers, i)
        
    return model


# In[12]:


def train(model, data, path_model, learning_rate = 0.001, epochs = 20, batch_size = 32, patience=10, verbose = verbose):
    """
    Compile and fit model for *epochs* rounds of training, dividing learning rate by 10 after each round
    
    Fitting will stop if val_acc does not improve for at least patience epochs
    
    Only the best weights will be kept
    
    Good practice is to decrease the learning rate by a factor of 10 after each plateau and train some more 
    (after first re-loading best weights from previous training round)...

    for example:
        fit_history = train(model, data, path_model = pwd+'models/', learning_rate = 0.001, epochs = 30)
        model.load_weights(path_model + "model.h5")
        fit_history = train(model, data, path_model = pwd+'models/', learning_rate = 0.0001, epochs = 30)
    
    :model: model object to train
    :data: data object
    :path_model: path to save fit logs and model snapshot
    :learning_rate: learning rate parameter for Adam optimizer (default is 0.001)
    
    :epochs: number of training epochs per fit round (subject to patience)
    :batch_size: number of samples in each batch
    :patience: how many epochs without val_acc improvement before stopping fit round
    :verbose: print progress
    
    """

    # TODO: refactor
    path_model = pwd + 'models/'
    
    # create optimizer with given learning rate 
    opt = Adam(lr = learning_rate)
    
    # compile model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # setup training callbacks
    callback_stopper = EarlyStopping(monitor='val_acc', patience=patience, verbose=0)
    callback_csvlogger = CSVLogger(path_model + 'training.log')
    callback_checkpointer = ModelCheckpoint(path_model +  'model.h5', monitor='val_acc', save_best_only=True, verbose=verbose)
    callback_tensorboard = TensorBoard(log_dir='tensorboard', histogram_freq=0, write_graph=True, write_images=True)
    # to start tensorboard, run the following from the package base directory: tensorboard --logdir tensorboard/
    callbacks = [callback_stopper, callback_checkpointer, callback_csvlogger, callback_tensorboard]
    
    return model.fit(data.x_train, data.y_train, 
              validation_data=(data.x_valid, data.y_valid),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              shuffle=True,
              verbose=verbose)


# In[ ]:


# fit_history = train(model, data, path_model = pwd+'models/', learning_rate = 0.001)
