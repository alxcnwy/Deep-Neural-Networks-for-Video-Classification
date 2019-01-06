#!/usr/bin/env python
# coding: utf-8

# In[1]:


### TODO:
# * add train/valid/test generators to data
# * need option to apply preprocessor when requesting frame data
# * make sure don't recompute sequences if inputs don't change


# In[2]:


# whether to log each feature and sequence status
verbose = True


# In[3]:


import os
import pandas as pd
import numpy as np
import json
from PIL import Image
import cv2
from sklearn.utils import shuffle
import sys
sys.path.append('..')


# In[4]:


# import pretrained model functions
from deepvideoclassification.models import precompute_CNN_features    
from deepvideoclassification.models import load_pretrained_model_preprocessor
from deepvideoclassification.models import load_pretrained_model

# import pretrained model properties
from deepvideoclassification.models import pretrained_model_len_features
from deepvideoclassification.models import pretrained_model_sizes
from deepvideoclassification.models import pretrained_model_names, poolings


# In[5]:


# setup paths
pwd = os.getcwd().replace("deepvideoclassification","")
path_cache = pwd + 'cache/'
path_data = pwd + 'data/'


# In[6]:


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


# In[7]:


# read vid folders
def get_video_paths():
    """
    Return list of video paths 

    Videos should be in /data/video_1/, /data/video_2/ style folders 
    with sequentially numbered frame images e.g. /data/video_1/frame00001.jpg

    There should be at least 3 videos, 1 for each of train/test/valid splits
    Split assignment is given in /data/labels.csv (also to be provided by user)

    Functionality to use different parts of a video as train/valid/test 
    not currently implemented.
    """
    path_videos = []
    for filename in os.listdir(path_data):
        if os.path.isdir(os.path.join(path_data, filename)):
            path_videos.append(filename)

    path_videos = [path_data + v + '/' for v in path_videos]

    # make sure that there is video data in /data/ and give instructions if not done correctly
    assert len(path_videos)>0, "There need to be at least 3 video folders (at least 1 for each of train, valid,     and test splits) in /data/ - each video should be its own folder of frame images with ascending time-ordered     filenames e.g. /data/vid1/frame00001.jpg ... videos assignment to train/valid/test split should be given in     /data/labels.csv ... cross-validation or train/valid/test splits within a single long video not currently implemented"

    return path_videos


# In[8]:


def resize_frames(target_size):
    """
    Resize the frames of all videos and save them to /cache/ 
    to make model fitting faster .

    We resize once upfront rather than each we use a pretrained model or architecture.

    Our models require inputs resized to:
    * 224 x 224 VGG16, ResNet50, DenseNet, MobileNet
    * 299 x 299 XCeption, InceptionV3, InceptionResNetV2
    * 112 x 112 3D CNN 
    """

    if not os.path.exists(path_cache + 'frames/' + str(target_size[0]) + "_" + str(target_size[1]) + '/'):

        # read vid paths
        path_videos = get_video_paths()

        # loop over all vids and resize frames, saving to new folder in /cache/frames/
        for c,path_video in enumerate(path_videos):

            logger.info("resizing vid {}/{} to {}x{}".format(c+1,len(path_videos),target_size[0], target_size[1]))

            # get vid name from path
            video_name = path_video.split("/")[-2]

            # create directory for resized frames - just storing arrays now so commented out
            # e.g. path_vid_resized = /cache/frames/224_224/s23-4847/
            # path_vid_resized = path_cache + 'frames/'
            # path_vid_resized += str(target_size[0]) + "_" + str(target_size[1]) + '/' 
            # path_vid_resized += video_name + '/'

            # load frame paths for vid
            path_frames = os.listdir(path_video)
            path_frames = [path_video + f for f in path_frames if f != '.DS_Store']
            path_frames.sort()

            # load frames
            frames = []
            for path_frame in path_frames:

                # open image and resize
                filename = path_frame.split("/").pop()
                img_frame = Image.open(path_frame)
                img_frame = img_frame.resize(target_size)
                # img_frame.save(path_vid_resized + filename, "JPEG", quality = 100)

                # convert to array and append to list
                img_frame = np.array(img_frame)
                frames.append(img_frame)

            # save array of resized frames
            np.save("/mnt/seals/cache/frames/" + str(target_size[0]) + "_" + str(target_size[1]) + "/" + video_name, np.array(frames))


# In[9]:


def get_labels():
    # read labels - should be CSV with columns "video","frame","label","split"
    # e.g. "s1-218", "s1-218-00001.jpeg", "noseal", "train"
    labels = None
    try:
        labels = pd.read_csv(path_data + 'labels.csv', usecols=['video','frame','label','split'])
    except ValueError as e:
        raise Exception("Labels file must contain columns ['video','frame','label','split'] - if you only have ['video','frame','label'], use the helper function add_splits_to_labels_file to add train/valid/test splits to your labels file")
    except FileNotFoundError as e:
        raise Exception("No labels found - please save labels file to /data/labels.csv") from e

    return labels.sort_values(["video","frame"])


# In[10]:


def create_video_label_arrays():
    """
    Create numpy array with labels for each vid and a label_map.json file
    in /cache/labels/
    """

    # create folder for labels
    if not os.path.exists(path_cache + 'labels/'):
        os.makedirs(path_cache + 'labels/')

    # load labels
    labels = get_labels()

    # build label_map
    label_dummies = pd.get_dummies(labels, columns = ['label'])

    # get label columns list and build label map dict
    label_columns = []
    label_map = {}
    label_map_idx = 0
    for i, col in enumerate(label_dummies.columns):
        if col[:6] == 'label_':
            label_columns.append(col)
            label_map[label_map_idx] = col
            label_map_idx+=1

    # save label map to json
    with open(path_cache + 'labels/label_map.json', 'w') as fp:
        json.dump(label_map, fp)

    # get video paths
    path_videos = get_video_paths()

    # save numpy array of labels for each vid
    for path_video in path_videos:

        # get vid name from path
        video_name = path_video.split("/")[-2]

        vid_labels = np.array(label_dummies[label_dummies['video'] == video_name][label_columns])

        # save labels array for this vid
        np.save("/mnt/seals/cache/labels/" + video_name, np.array(vid_labels))


# In[11]:


def load_label_map():
    """
    Returns label map - read from disk
    """

    # load label map from disk
    label_map = None
    try:
        if os.path.exists(path_cache + 'labels/label_map.json'):
            with open(path_cache + 'labels/label_map.json', 'r') as fp:
                label_map = json.load(fp)
        else:
            # build labels and label map
            create_video_label_arrays()
            if os.path.exists(path_cache + 'labels/label_map.json'):
                with open(path_cache + 'labels/label_map.json', 'r') as fp:
                    label_map = json.load(fp)
    except Exception as e:
        logger.error ('label map not found - make sure /data/labels.csv exists and data cache has been built')

    return label_map


# In[34]:


class Data(object):
    
    def __init__(self, sequence_length, 
                 return_CNN_features = False, pretrained_model_name = None, pooling = None, 
                 frame_size = None, augmentation = False, oversampling = False):
        """
        Data object constructor
        
        
        :sequence_length: number of frames in sequence to be returned by Data object
        :return_CNN_features: whether to return precomputed features or return frames (or sequences of features/frames if sequence_length>1)

        :return_features: if True then return features (or sequences of feature) from pretrained model, if False then return frames (or sequences of frames)        
        :pretrained_model_name: name of pretrained model (or None if not using pretrained model e.g. for 3D-CNN)
        :pooling: name of pooling variant (or None if not using pretrained model e.g. for 3D-CNN)
        :frame_size: size that frames are resized to (this is looked up for pretrained models)
        :augmentation: whether to apply data augmentation (horizontal flips)
        :oversampling: whether to apply oversampling to create class balance
        
        
        Notes: 
        * if pretrained_model_name != None and return_CNN_features=False then will first apply preprocessor to frames (or frame sequences)
        """
    
        # required params
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        
        # optional params
        self.pretrained_model_name = pretrained_model_name
        self.pooling = pooling
        self.return_CNN_features = return_CNN_features
        self.augmentation = augmentation
        self.oversampling = oversampling
        
        # init model data
        self.x_train = []
        self.y_train = []
        #
        self.x_valid = []
        self.y_valid = []
        # 
        self.x_test = []
        self.y_test = []
        
        # fix case sensitivity
        if type(self.pretrained_model_name) == str:
            self.pretrained_model_name = self.pretrained_model_name.lower()
        if type(self.pooling) == str:
            self.pooling = self.pooling.lower()
        
        ################
        ### Prepare data
        ################
        
        # get video paths
        self.path_videos = get_video_paths()
        
        # create label array for each video and load label map
        create_video_label_arrays()
        self.label_map = load_label_map()
        
        # get labels
        self.labels = get_labels()
        
        # create dict mapping video to train/valid/test split assignment
        video_splits = self.labels[['video','split']].drop_duplicates()
        video_splits.set_index("video", inplace=True)
        video_splits = video_splits.to_dict()['split']
        self.video_splits = video_splits
        
        # look up target size for pretrained model
        if pretrained_model_name is not None:
            self.frame_size = pretrained_model_sizes[pretrained_model_name]
        
        # precompute resized frames (won't recompute if already resized)
        resize_frames(self.frame_size)

        # pre compute CNN features (won't recompute if already computed)
        if return_CNN_features and pretrained_model_name is not None:
            precompute_CNN_features(self.pretrained_model_name, self.pooling)
            
        # get preprocessor given pretrained if we will need to apply preprocessor 
        # (i.e. if return_CNN_features == False and pretrained_model_name != None)
        if not return_CNN_features and pretrained_model_name is not None:
            self.preprocess_input = load_pretrained_model_preprocessor(self.pretrained_model_name)
        
        
        ###################################
        ### load features / build sequences
        ###################################
        
        
        # load features/frames from all videos and concat into big array for each of train, valid and test
        if self.sequence_length > 1:
            
            ###################
            ### frame sequences
            ###################
            
            if return_CNN_features:
                
                #####################
                ### feature sequences
                #####################
                
                path_features = path_cache + 'features/' + pretrained_model_name + "/" + pooling + '/'
                path_labels = path_cache + 'labels/'
                
                # read vid paths
                path_videos = get_video_paths()

                # loop over all vids and resize frames, saving to new folder in /cache/frames/
                for c, path_video in enumerate(path_videos):

                    # get vid name from path
                    video_name = path_video.split("/")[-2]
                    
                    
                    ### create sequence: features
                    # load precomputed features
                    features = np.load(path_features + video_name + '.npy')
                    # build sequences
                    x = []
                    for i in range(sequence_length, len(features) + 1):
                        x.append(features[i-sequence_length:i])
                    x = np.array(x)
                    
                    
                    ### create sequence: labels
                    # load precomputed labels
                    labels = np.load(path_labels + video_name + '.npy')     

                    # temp lists to store sequences
                    y = []
                    for i in range(sequence_length, len(labels) + 1):
                        y.append(labels[i-1])
                    y = (np.array(y))

                    
                    ### build output
                    if self.video_splits[video_name] == "train":
                        self.x_train.append(x)
                        self.y_train.append(y)
                    if self.video_splits[video_name] == "valid":
                        self.x_valid.append(x)
                        self.y_valid.append(y)
                    if self.video_splits[video_name] == "test":
                        self.x_test.append(x)
                        self.y_test.append(y)
                        
            else:

                ###################
                ### frame sequences
                ###################
                
                # load resized numpy array
                path_vid_resized = path_cache + 'frames/'
                path_vid_resized += str(self.frame_size[0]) + "_" + str(self.frame_size[1]) + '/' 
                
                path_labels = path_cache + 'labels/'
                
                # read vid paths
                path_videos = get_video_paths()

                # loop over all vids and resize frames, saving to new folder in /cache/frames/
                for c, path_video in enumerate(path_videos):

                    # get vid name from path
                    video_name = path_video.split("/")[-2]
                    
                    ### create sequence: features
                    # load precomputed frames
                    frames = np.load(path_vid_resized  + video_name + '.npy')
                    
                    # first apply preprocessing if pretrained model given
                    if pretrained_model_name != None:
                        frames = self.preprocess_input(frames)
                    
                    # build sequences
                    x = []
                    for i in range(sequence_length, len(frames) + 1):
                        x.append(frames[i-sequence_length:i])
                    x = np.array(x)
                    
                    
                    ### create sequence: labels
                    # load precomputed labels
                    labels = np.load(path_labels + video_name + '.npy')     

                    # temp lists to store sequences
                    y = []
                    for i in range(sequence_length, len(labels) + 1):
                        y.append(labels[i-1])
                    y = (np.array(y))

                    ### build output
                    if self.video_splits[video_name] == "train":
                        self.x_train.append(x)
                        self.y_train.append(y)
                    if self.video_splits[video_name] == "valid":
                        self.x_valid.append(x)
                        self.y_valid.append(y)
                    if self.video_splits[video_name] == "test":
                        self.x_test.append(x)
                        self.y_test.append(y)
                
        else:
            ###############
            ### no sequence
            ###############
            if return_CNN_features:
                
                ###################
                ### feature vectors
                ###################
                
                path_features = path_cache + 'features/' + pretrained_model_name + "/" + pooling + '/'
                path_labels = path_cache + 'labels/'
                
                # read vid paths
                path_videos = get_video_paths()

                # loop over all vids and resize frames, saving to new folder in /cache/frames/
                for c, path_video in enumerate(path_videos):

                    # get vid name from path
                    video_name = path_video.split("/")[-2]
                    
                    ### load precomputed features
                    x = np.load(path_features + video_name + '.npy')
                    y = np.load(path_labels + video_name + '.npy')
                    
                    ### build output
                    if self.video_splits[video_name] == "train":
                        self.x_train.append(x)
                        self.y_train.append(y)
                    if self.video_splits[video_name] == "valid":
                        self.x_valid.append(x)
                        self.y_valid.append(y)
                    if self.video_splits[video_name] == "test":
                        self.x_test.append(x)
                        self.y_test.append(y)
            else:
                
                #################
                ### single frames
                #################
                
                # load resized numpy array
                path_vid_resized = path_cache + 'frames/'
                path_vid_resized += str(self.frame_size[0]) + "_" + str(self.frame_size[1]) + '/' 
                
                path_labels = path_cache + 'labels/'
                
                # read vid paths
                path_videos = get_video_paths()

                # loop over all vids and resize frames, saving to new folder in /cache/frames/
                for c, path_video in enumerate(path_videos):

                    # get vid name from path
                    video_name = path_video.split("/")[-2]
                    
                    # load precomputed numpy arrays for frames and labels
                    x = np.load(path_vid_resized  + video_name + '.npy')
                    y = np.load(path_labels + video_name + '.npy')
                    
                    # apply preprocessing if pretrained model given
                    if pretrained_model_name != None:
                        x = self.preprocess_input(x)
                
                    ### build output
                    if self.video_splits[video_name] == "train":
                        self.x_train.append(x)
                        self.y_train.append(y)
                    if self.video_splits[video_name] == "valid":
                        self.x_valid.append(x)
                        self.y_valid.append(y)
                    if self.video_splits[video_name] == "test":
                        self.x_test.append(x)
                        self.y_test.append(y)
            
            
            
        ########################
        ### reshape list outputs
        ########################
        ## e.g. (9846, 224, 224, 3) for frames [return_CNN_features=True]
        ## or  (9846, 512) for features [return_CNN_features=False]
        self.x_train = np.concatenate(self.x_train, axis=0)
        self.y_train = np.concatenate(self.y_train, axis=0)
        self.x_valid = np.concatenate(self.x_valid, axis=0)
        self.y_valid = np.concatenate(self.y_valid, axis=0)
        self.x_test = np.concatenate(self.x_test, axis=0)
        self.y_test = np.concatenate(self.y_test, axis=0)
        
        
        
        #################################
        ### get file paths for each split
        #################################
        # get file paths: train
        dflab = self.labels[self.labels['split'] == 'train']
        self.paths_train = list(path_data + dflab['video'] + "/" + dflab['frame'])

        # get file paths: valid
        dflab = self.labels[self.labels['split'] == 'valid']
        self.paths_valid = list(path_data + dflab['video'] + "/" + dflab['frame'])

        # get file paths: test
        dflab = self.labels[self.labels['split'] == 'test']
        self.paths_test = list(path_data + dflab['video'] + "/" + dflab['frame'])
        
        # pull number of classes from labels shape
        self.num_classes = self.y_train.shape[1]
            

    def __str__(self):
        return "x_train: {}, y_train: {} ... x_valid: {}, y_valid: {} ... x_test: {}, y_test: {}".format(self.x_train.shape,self.y_train.shape,self.x_valid.shape,self.y_valid.shape,self.x_test.shape,self.y_test.shape)
            
    def shuffle(self):
        """
        randomize the order of samples in train and valid splits
        """
        ###########
        ### shuffle
        ###########
        # paths will no longer be correct (they're for debugging anyway)
        self.x_train, self.y_train, self.paths_train = shuffle(self.x_train, self.y_train, self.paths_train)
        self.x_valid, self.y_valid, self.paths_valid = shuffle(self.x_valid, self.y_valid, self.paths_valid)


# In[35]:


# data = Data(sequence_length = 1, 
#             return_CNN_features = True, 
#             pretrained_model_name='vgg16', 
#             pooling='max')

