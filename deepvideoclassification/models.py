#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### TODO
# * check base model outputs without pooling
# * edit image MLPs to assume sequence_length of 1


# In[1]:


# whether to log each feature and sequence status
verbose = True


# In[2]:


import os
import pandas as pd
import numpy as np
from PIL import Image
import json
import cv2


# In[ ]:


from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, Convolution1D, Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop


# In[3]:


# setup paths
pwd = os.getcwd().replace("deepvideoclassification","")
path_cache = pwd + 'cache/'
path_data = pwd + 'data/'


# In[4]:


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

# In[4]:


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


# In[5]:


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


# In[6]:


pretrained_model_names = ["inception_resnet_v2", "inception_v3", "mobilenetv2_1.00_224", "resnet50", "vgg16", "xception"]
poolings = ['max','avg']


# In[ ]:


def load_pretrained_model(pretrained_model_name, pooling):
    """ Load pretrained model with given pooling applied
    
    Args:
        pretrained_model: name of pretrained model ["Xception", "VGG16", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNetV2"]
        pooling: pooling strategy for final pretrained model layer ["max","avg"]
    
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


# In[ ]:


def precompute_CNN_features(pretrained_model_name, pooling):
    """ Save pretrained features array computed over all frames of each video using given pretrained model and pooling method
    
    Args:
        pretrained_model_name: pretrained model object loaded using `load_pretrained_model`
        pooling: pooling method used with pretrained model
    
    Returns:
        None. Saves pretrained frames to `/cache/features/*pretrained_model_name*/*pooling*/*video_name*.npy`
    """
    
    pretrained_model_name = pretrained_model_name.lower()
    
    # setup path to save features
    path_features = path_cache + 'features/' + pretrained_model_name + "/" + pooling + '/'
    
    if not os.path.exists(path_features):
        
        os.makedirs(path_features)
        
        # load pretrained model
        pretrained_model = load_pretrained_model(pretrained_model_name, pooling)

        # load preprocessing function
        preprocess_input = load_pretrained_model_preprocessor(pretrained_model_name)

        # lookup pretrained model input shape
        model_input_size = pretrained_model_sizes[pretrained_model_name]
        
        # precompute features for each video using pretrained model
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


# # Image/video classification model object with various architectures

# In[11]:


class Model(object):
    
    def __init__(self, sequence_length, num_classes, target_size, 
                architecture, pretrained_model_name, pooling = None,
                sequence_model = None, sequence_model_layers = 1,
                layer_1_size = 0, layer_2_size = 0, layer_3_size = 0, 
                dropout = 0):
        """
        Model object constructor
        
        :sequence_length: number of frames in sequence to be returned by Data object
        :num_classes: number of classes to predict
        :target_size: size that frames are resized to (different models / architectures accept different input sizes)

        :architecture: architecture of model in [image_MLP_frozen, image_MLP_trainable, video_MLP_concat, video_LRCNN_frozen, video_LRCNN_trainable, 3DCNN]
        :pretrained_model_name: name of pretrained model (or None if not using pretrained model e.g. for 3D-CNN)
        :pooling: name of pooling variant (or None if not using pretrained model e.g. for 3D-CNN or if fitting more non-dense layers on top of pretrained model base)
        
        :sequence_model: sequence model in [LSTM, SimpleRNN, GRU, Convolution1D]
        :sequence_model_layers: default to 1, can be stacked 2 or 3 layer sequence model (assume always stacking the same sequence model, not mixing LSTM and GRU, for example)
        
        :layer_1_size: number of neurons in layer 1
        :layer_2_size: number of neurons in layer 2
        :layer_3_size: number of neurons in layer 3
        
        :dropout: amount of dropout to add (same amount throughout)
        """
    
        # required params
        self.sequence_length = sequence_length
        self.target_size = target_size
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
        self.dropout = dropout
        
        # maybe read target size from pretrained model 
        if pretrained_model_name is not None:
            self.num_features = pretrained_model_len_features[pretrained_model_name]
        
        self.model = None
        
        if architecture == "image_MLP_frozen":
            
            ####################
            ### image_MLP_frozen
            ####################
            
            # image classification (single frame)
            # train MLP on top of weights extracted from pretrained CNN with no fine-tuning
            
            model = Sequential()
            
            # 1st layer group
            if layer_1_size > 0:
                model.add(Flatten(input_shape=(self.sequence_length, self.num_features)))
                model.add(Dense(layer_1_size, activation='relu'))
            else:
                # flatten input
                model.add(Flatten(input_shape=(self.sequence_length, self.num_features)))

            if layer_2_size > 0 and layer_1_size > 0:
                model.add(Dense(layer_2_size, activation='relu'))
                if dropout > 0:
                    model.add(Dropout(dropout))

            if layer_2_size > 0 and layer_3_size > 0 and layer_1_size > 0:
                model.add(Dense(layer_3_size, activation='relu'))
                if dropout > 0:
                    model.add(Dropout(dropout))
                
            # final layer
            model.add(Dense(NUM_CLASSES, activation='softmax'))

        elif architecture == "image_MLP_trainable":
            
            #######################
            ### image_MLP_trainable
            #######################
            
            # image classification (single frame)
            # fine-tune pretrained CNN and fit MLP on top
            #
            # later we will compare our best fine-tuned CNN as a feature extractor vs fixed CNN features
            
            # create the base pre-trained model
#             base_model = InceptionV3(weights='imagenet', include_top=False)
            base_model = load_pretrained_model(pretrained_model_name, pooling)
            
        
        
            ###### TODO

            
            
            
            # add a global spatial average pooling layer
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            # let's add a fully-connected layer
            x = Dense(1024, activation='relu')(x)
            # and a logistic layer -- let's say we have 200 classes
            predictions = Dense(200, activation='softmax')(x)

            # this is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)

            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False

                
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
            # note: no fine-tuning of CNN
            
            if self.sequence_model == "LSTM":
                print("TODO")
#                 sequence_model_layers


            elif self.sequence_model == "SimpleRNN":
                print("TODO")
            elif self.sequence_model == "GRU":
                print("TODO")
            elif self.sequence_model == "Convolution1D":
                print("TODO")
            else:
                raise NameError('Invalid sequence_model - must be one of [LSTM, SimpleRNN, GRU, Convolution1D]')    

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
            
            print('TODO')
            
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
            model.add(Dense(487, activation='softmax', name='fc8'))
            
        else:
            raise NameError('Invalid architecture - must be one of [image_MLP_frozen, image_MLP_trainable, video_MLP_concat, video_LRCNN_frozen, video_LRCNN_trainable, 3DCNN]')    
        
        # set class model to constructed model
        self.model = model


# # Move to experiment

# In[ ]:


# # define optimizer and compile model
# # (compiling the model should be done *after* setting layers to non-trainable)
# opt = Adam()
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[16]:


# def update_learning_rate(multiplier = 0.1):
#     """Update learning rate by multiplier"""
#     K.set_value(model.optimizer.lr, multiplier * model.optimizer.lr.get_value())

