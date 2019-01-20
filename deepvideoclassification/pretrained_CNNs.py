#!/usr/bin/env python
# coding: utf-8

# In[1]:


# whether to log each feature and sequence status
verbose = True


# In[2]:


import os
import sys
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import itertools
from contextlib import redirect_stdout
sys.path.append('..')


# In[3]:


from keras.preprocessing.image import img_to_array


# In[4]:


# setup paths
pwd = os.getcwd().replace("deepvideoclassification","")
path_cache = pwd + 'cache/'
path_data = pwd + 'data/'


# In[5]:


# setup logging
# any explicit log messages or uncaught errors to stdout and file /logs.log
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(pwd, "logs")),
        logging.StreamHandler()
    ])
# init logger
logger = logging.getLogger()
# make logger aware of any uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_exception


# # Pretrained_CNNs

# In[6]:


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


# In[7]:


# Notice the output features of the models group (roughly) into 3 buckets: 512, 1280 (with 1536), and 2048... 
# we'll do a grid search for 1 model in each bucket then run the best model in the grid for all other models in that bucket
pretrained_model_names_bucketed = ['inception_resnet_v2', 'vgg16', 'resnet50']


# In[8]:


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


# In[9]:


pretrained_model_names = ["inception_resnet_v2", "inception_v3", "mobilenetv2_1.00_224", "resnet50", "vgg16", "xception"]
poolings = ['max','avg']


# In[10]:


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


# In[11]:


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


# In[12]:


def precompute_CNN_features(pretrained_model_name, pooling, model_weights_path = None, custom_model_name = None):
    """ 
    Save pretrained features array computed over all frames of each video 
    using given pretrained model and pooling method
    
    :pretrained_model_name: pretrained model object loaded using `load_pretrained_model`
    :pooling: pooling method used with pretrained model
    :model_weights_path: path to custom model weights if we want to load CNN model we've fine-tuned to produce features (e.g. for LRCNN)
    :custom_model_name: custom output name to append to pretrained model name

    """
    
    assert (pretrained_model_name is not None or custom_model_name is not None), "need to specify a pretrained_model_name in ['Xception', 'VGG16', 'ResNet50', 'InceptionV3', 'InceptionResNetV2', 'MobileNetV2'] or a custom_model_name"
    
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

