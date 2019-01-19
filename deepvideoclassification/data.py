#!/usr/bin/env python
# coding: utf-8

# In[5]:


# whether to log each feature and sequence status
verbose = 1


# In[6]:


import os
import pandas as pd
import numpy as np
import json
from PIL import Image
import cv2
from sklearn.utils import shuffle
import sys
sys.path.append('..')

import h5py
import random


# In[7]:


import keras


# In[8]:


# import pretrained model functions
from deepvideoclassification.models import precompute_CNN_features
from deepvideoclassification.models import load_pretrained_model_preprocessor
from deepvideoclassification.models import load_pretrained_model

# import pretrained model properties
from deepvideoclassification.models import pretrained_model_len_features
from deepvideoclassification.models import pretrained_model_sizes
from deepvideoclassification.models import pretrained_model_names, poolings


# In[9]:


# setup paths
pwd = os.getcwd().replace("deepvideoclassification","")
path_cache = pwd + 'cache/'
path_data = pwd + 'data/'


# In[10]:


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


# In[11]:


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


# In[12]:


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

    assert (target_size is not None), "target_size (or pretrained_model_name which implies a target_size) must be specified"
    
    if not os.path.exists(path_cache + 'frames/' + str(target_size[0]) + "_" + str(target_size[1]) + '/'):
        
        os.makedirs(path_cache + 'frames/' + str(target_size[0]) + "_" + str(target_size[1]) + '/')

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
            np.save(path_cache + "frames/" + str(target_size[0]) + "_" + str(target_size[1]) + "/" + video_name, np.array(frames))


# In[13]:


def get_labels():
    # read labels - should be CSV with columns "video","frame","label","split"
    # e.g. "s1-218", "s1-218-00001.jpeg", "noseal", "train"
    labels = None
    try:
        labels = pd.read_csv(path_data + 'labels.csv', usecols=['video','frame','label','split'])
    except ValueError as e:
        raise Exception("Labels file must contain columns ['video','frame','label','split'] - if you only have ['video','frame','label'], use Jupyter notebook in notebooks/add_splits_to_labels_file.ipynb to add train/valid/test splits to your labels file")
    except FileNotFoundError as e:
        raise Exception("No labels found - please save labels file to /data/labels.csv") from e

    return labels.sort_values(["video","frame"])


# In[14]:


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
        np.save(path_cache + "/labels/" + video_name, np.array(vid_labels))


# In[15]:


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


# In[16]:


class DataGenerator(keras.utils.Sequence):
    """
    Generator (used in Data) that generates data for Keras fit_generator method because full dataset too big to load into memory
    
    > inherits from keras.utils.Sequence
    
    """
    def __init__(self, batch_size, h5_path, h5_row_count):
        """
        Initialization DataGenerator class
        
        :batch_size: number of samples to return in batch 
        :h5_path: path to h5 dataset (where we stored the generated sequence data via save_frame_sequences_to_h5())
        :h5_row_count: number of rows in h5 dataset
        
        """
        self.batch_size = batch_size
        self.h5_path = h5_path
        self.h5_row_count = h5_row_count
        
        # init (shuffle dataset)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.h5_row_count / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        x, y = self.__data_generation(batch_indexes)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.h5_row_count)
        
        # shuffle indexes -> shuffle samples returned in each batch
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        """
        Generates data containing batch_size samples
        
        :batch_indexes: list (of size batch_size) with indexes into h5 file
        """ 
        x, y = None, None

        # slices into h5 file need to be sorted
        batch_indexes.sort()

        # read sample from h5 file
        with h5py.File(self.h5_path, 'r') as h5:
            ### read sample indexes from h5 file
            # sample sequences
            x = h5['sequences'][batch_indexes,:]
            # sample labels
            y = h5['labels'][batch_indexes,:]

        return x, y 


# In[27]:


class Data(object):
    
    def __init__(self, sequence_length, 
                    return_CNN_features = False, pretrained_model_name = None, pooling = None, 
                    frame_size = None, augmentation = False, oversampling = False,
                    model_weights_path = None, custom_model_name = None,
                    return_generator = False, batch_size = None,
                    verbose = True):
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
        
        :model_weights_path: path to custom model weights if we want to load CNN model we've fine-tuned to produce features (e.g. for LRCNN)
        :custom_model_name: custom output name to append to pretrained model name
        
        :return_generator: if True and sequence_length > 1 and return_CNN_features == False, then do not return dataset, instead construct h5 file with sequences for each split and return generator that samples from that (dataset of sequecne frames too big to load into memory)
        :batch_size: size of batches that generator must return
        
        :verbose: whether to log details
        
        Notes: 
        * if pretrained_model_name != None and return_CNN_features=False then will first apply preprocessor to frames (or frame sequences)
        * if return_generator = True and sequence_length > 1 and return_CNN_features == False, large h5 files will be created in cache before returning generator
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
        self.model_weights_path = model_weights_path
        
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
        
        # check that there is 1 frame file for each label file and raise error if they don't match
        paths_frames = []
        for folder, subs, files in os.walk(path_data):        
            for filename in files:
                if filename[-4:].lower() == '.jpg' or filename[-4:].lower() == 'jpeg' or filename[-4:].lower() == 'png':
                    paths_frames.append(os.path.abspath(os.path.join(folder, filename)))
        if len(paths_frames) != len(self.labels):
            error_msg = 'IMPORTANT ERROR: Number of frames ({}) in /data/ video folders needs to match number of labels ({}) in labels.csv - use notebooks/helper_check_frames_against_labels.ipynb to investigate... Note, only labels.csv and the frames you want to use (in video subfolders) should be in /data/'.format(len(paths_frames), len(self.labels))
            logger.info(error_msg)
            raise ValueError(error_msg)
        
        # pull number of classes from labels shape
        self.num_classes = self.labels['label'].nunique()
        
        # create dict mapping video to train/valid/test split assignment
        video_splits = self.labels[['video','split']].drop_duplicates()
        video_splits.set_index("video", inplace=True)
        video_splits = video_splits.to_dict()['split']
        self.video_splits = video_splits
        
        # look up target size for pretrained model
        if pretrained_model_name is not None:
            self.frame_size = pretrained_model_sizes[self.pretrained_model_name]
        
        # precompute resized frames (won't recompute if already resized)
        resize_frames(self.frame_size)

        # pre compute CNN features (won't recompute if already computed)
        if self.return_CNN_features and self.pretrained_model_name is not None:
            # check if pass custom weights to precompute from
            if self.model_weights_path is not None and self.custom_model_name is not None:
                # precompute with custom weights input and name
                precompute_CNN_features(self.pretrained_model_name, self.pooling, self.model_weights_path, self.custom_model_name)
            else:
                precompute_CNN_features(self.pretrained_model_name, self.pooling)
            
            
            
        # get preprocessor given pretrained if we will need to apply preprocessor 
        # (i.e. if return_CNN_features == False and pretrained_model_name != None)
        if not self.return_CNN_features and self.pretrained_model_name is not None:
            self.preprocess_input = load_pretrained_model_preprocessor(self.pretrained_model_name)
        
        
        self.verbose = verbose
        
        self.return_generator = return_generator
        self.batch_size = batch_size
        
        # do some checks
        if self.return_generator:
            assert self.batch_size != None, "batch size required to construct generator"
        if self.return_generator:
            assert self.return_CNN_features == False, "generator only implemented for frame sequences - features usually large enough to load into memory [may take a few minutes]"
        
        ###################################
        ### load features / build sequences
        ###################################
        
        
        # load features/frames from all videos and concat into big array for each of train, valid and test
        if self.sequence_length > 1:
            
            ### sequences
            
            if return_CNN_features:
                
                if verbose:
                    logging.info("Loading features sequence data into memory [may take a few minutes]")
                
                #####################
                ### feature sequences
                #####################
                
                path_features = path_cache + 'features/' + self.pretrained_model_name + "/" + self.pooling + '/'
                if not self.return_CNN_features and self.pretrained_model_name is not None:
                    path_features = path_cache + 'features/' + self.pretrained_model_name + "__" + self.custom_model_name + "/" + self.pooling + '/'
                path_labels = path_cache + 'labels/'
                
                # read vid paths
                path_videos = get_video_paths()

                # loop over all vids and load precomputed features into memory as sequences
                for c, path_video in enumerate(path_videos):

                    if verbose:
                        logging.info("Loading features sequence data into memory {}/{}".format(c+1,len(path_videos)))
                    
                    # get vid name from path
                    video_name = path_video.split("/")[-2]
                    
                    ### create sequence: features
                    # load precomputed features
                    features = np.load(path_features + video_name + '.npy')
                    # build sequences
                    x = []
                    for i in range(self.sequence_length, len(features) + 1):
                        x.append(features[i-sequence_length:i])
                    x = np.array(x)
                    
                    
                    ### create sequence: labels
                    # load precomputed labels
                    labels = np.load(path_labels + video_name + '.npy')     

                    # temp lists to store sequences
                    y = []
                    for i in range(self.sequence_length, len(labels) + 1):
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
                
                # load full frame sequecne dataset into memory and return
                if not return_generator:
                    
                    ##############################################################################
                    ### load full sequence dataset into memory (will likely run into memory error)
                    ##############################################################################
                    
                    if verbose:
                        logging.info("Loading frame sequence data into memory [may take a few minutes]")

                    # load resized numpy array
                    path_vid_resized = path_cache + 'frames/'
                    path_vid_resized += str(self.frame_size[0]) + "_" + str(self.frame_size[1]) + '/' 

                    path_labels = path_cache + 'labels/'

                    # read vid paths
                    path_videos = get_video_paths()

                    # loop over all vids and load full frame sequences into memory 
                    # (recommend using generator if lots of data to avoid out of memory issues)
                    for c, path_video in enumerate(path_videos):
                        
                        if verbose:
                            logging.info("Loading frame sequence data into memory {}/{}".format(c+1,len(path_videos)))

                        # get vid name from path
                        video_name = path_video.split("/")[-2]

                        ### create sequence: features
                        # load precomputed frames
                        frames = np.load(path_vid_resized  + video_name + '.npy')

                        # first apply preprocessing if pretrained model given
                        if self.pretrained_model_name != None:
                            frames = self.preprocess_input(frames.astype(np.float32))

                        # build sequences
                        x = []
                        for i in range(self.sequence_length, len(frames) + 1):
                            x.append(frames[i-sequence_length:i])
                        x = np.array(x)


                        ### create sequence: labels
                        # load precomputed labels
                        labels = np.load(path_labels + video_name + '.npy')     

                        # temp lists to store sequences
                        y = []
                        for i in range(self.sequence_length, len(labels) + 1):
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
                    #############################
                    ### Build sequences generator
                    #############################
            
                    # compute and save h5 sequence files (save_frame_sequences_to_h5 returns 
                    # the sequence sizes which we need for our generator)
                    self.total_rows_train, self.total_rows_valid, self.total_rows_test = self.save_frame_sequences_to_h5()
                    
                    # init generators
                    self.generator_train = DataGenerator(self.batch_size, self.path_h5_train, self.total_rows_train)
                    self.generator_valid = DataGenerator(self.batch_size, self.path_h5_valid, self.total_rows_valid)
                    self.generator_test = DataGenerator(self.batch_size, self.path_h5_test, self.total_rows_test)
                
        else:

            ### not sequence
            
            if return_CNN_features:
                
                if verbose:
                    logging.info("Loading features data into memory [may take a few minutes]")
                
                ###################
                ### feature vectors
                ###################
                
                path_features = path_cache + 'features/' + self.pretrained_model_name + "/" + self.pooling + '/'
                if not self.return_CNN_features and self.pretrained_model_name is not None:
                    path_features = path_cache + 'features/' + self.pretrained_model_name + "__" + self.custom_model_name + "/" + self.pooling + '/'
                
                path_labels = path_cache + 'labels/'
                
                # read vid paths
                path_videos = get_video_paths()

                # loop over all vids and load precomputed features
                for c, path_video in enumerate(path_videos):

                    if verbose:
                        logging.info("Loading features data into memory: {}/{}".format(c+1,len(path_videos)))
                    
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
                
                if verbose:
                    logging.info("Loading frames into memory [may take a few minutes]")
                
                #################
                ### single frames
                #################
                
                if not return_generator:
                    # load resized numpy array
                    path_vid_resized = path_cache + 'frames/'
                    path_vid_resized += str(self.frame_size[0]) + "_" + str(self.frame_size[1]) + '/' 

                    path_labels = path_cache + 'labels/'

                    # read vid paths
                    path_videos = get_video_paths()

                    # loop over all vids and load frames into memory
                    for c, path_video in enumerate(path_videos):

                        if verbose:
                            logging.info("Loading frames into memory: {}/{}".format(c+1,len(path_videos)))

                        # get vid name from path
                        video_name = path_video.split("/")[-2]

                        # load precomputed numpy arrays for frames and labels
                        x = np.load(path_vid_resized  + video_name + '.npy')
                        y = np.load(path_labels + video_name + '.npy')

                        # apply preprocessing if pretrained model given
                        if self.pretrained_model_name != None:
                            x = self.preprocess_input(x.astype(np.float32))

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
                    #############################
                    ### Build sequences generator
                    #############################
            
                    # compute and save h5 sequence files (save_frame_sequences_to_h5 returns 
                    # the sequence sizes which we need for our generator)
                    self.total_rows_train, self.total_rows_valid, self.total_rows_test = self.save_frame_sequences_to_h5()
                    
                    # init generators
                    self.generator_train = DataGenerator(self.batch_size, self.path_h5_train, self.total_rows_train)
                    self.generator_valid = DataGenerator(self.batch_size, self.path_h5_valid, self.total_rows_valid)
                    self.generator_test = DataGenerator(self.batch_size, self.path_h5_test, self.total_rows_test)
            
            
        #################################
        ### get file paths for each split
        #################################
        #
        # Note: only makes sense for sequence_length = 1
        
        # get file paths: train
        dflab = self.labels[self.labels['split'] == 'train']
        self.paths_train = list(path_data + dflab['video'] + "/" + dflab['frame'])

        # get file paths: valid
        dflab = self.labels[self.labels['split'] == 'valid']
        self.paths_valid = list(path_data + dflab['video'] + "/" + dflab['frame'])

        # get file paths: test
        dflab = self.labels[self.labels['split'] == 'test']
        self.paths_test = list(path_data + dflab['video'] + "/" + dflab['frame'])
        
        
        
        #################################################
        ### reshape list outputs (if not using generator)
        #################################################
        
        if not self.return_generator:
            ## e.g. (9846, 224, 224, 3) for frames [return_CNN_features=True]
            ## or  (9846, 512) for features [return_CNN_features=False]
            self.x_train = np.concatenate(self.x_train, axis=0)
            self.y_train = np.concatenate(self.y_train, axis=0)
            self.x_valid = np.concatenate(self.x_valid, axis=0)
            self.y_valid = np.concatenate(self.y_valid, axis=0)
            self.x_test = np.concatenate(self.x_test, axis=0)
            self.y_test = np.concatenate(self.y_test, axis=0)
            
            # shuffle train and validation set
            self.shuffle()
            

    def __str__(self):
        return "x_train: {}, y_train: {} ... x_valid: {}, y_valid: {} ... x_test: {}, y_test: {}".format(self.x_train.shape,self.y_train.shape,self.x_valid.shape,self.y_valid.shape,self.x_test.shape,self.y_test.shape)
            
    def shuffle(self):
        """
        Randomize the order of samples in train and valid splits
        """
        ###########
        ### shuffle
        ###########
        if self.sequence_length == 1:
            self.x_train, self.y_train, self.paths_train = shuffle(self.x_train, self.y_train, self.paths_train)
            self.x_valid, self.y_valid, self.paths_valid = shuffle(self.x_valid, self.y_valid, self.paths_valid)
        else:
            self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
            self.x_valid, self.y_valid = shuffle(self.x_valid, self.y_valid)

        

    # Even at small sequence lengths, loading the full dataset as 
    # a sequence into memory is not feasible so we need to use generators
    # that iterate over the dataset without loading it all into memory
    # 
    # For now, we will assume that we will load the features datasets into memory
    # because this is more feasible but for large datasets, we'd want to use generators for that too. 
    # An implementation for a feature generator  can be done by pattern matching the implementation for frames 
    # 
    # we first precompute a sequences h5 file (it's too big to fit in memory but we never have more than 1
    # video's sequences in memory) ...then we will initialize a generator that samples sequences from the 
    # h5 file and returns batches that will be passed to our model's fit_generator method

    def save_frame_sequences_to_h5(self):
        """
        Save sequence of frames to h5 files (1 for each split) in cache 
        because dataset too big to load into memory
        
        Will create generator that reads random rows from these h5 files
        
        Inspired by: https://stackoverflow.com/questions/41849649/write-to-hdf5-and-shuffle-big-arrays-of-data
        """
    
        #######################
        ### setup h5 file paths
        #######################
        
        if not os.path.exists(path_cache + 'sequences/'):
            os.makedirs(path_cache + 'sequences/')

        path_h5_base = path_cache + 'sequences/'

        # store h5 files in subfolder in cache/sequences/ either with pretrained model name or resize name
        # since we need to run preprocessing for pretrained models but not for vanilla resizing (3DCNN)
        if self.pretrained_model_name is not None:
            path_h5_base += self.pretrained_model_name + '/'
        else:
            path_h5_base += str(self.frame_size[0]) + "_" + str(self.frame_size[1]) + '/' 

        if not os.path.exists(path_h5_base):
            os.makedirs(path_h5_base)

        self.path_h5_train = path_h5_base + 'h5_' + str(self.sequence_length) + '_train.h5'
        self.path_h5_valid = path_h5_base + 'h5_' + str(self.sequence_length) + '_valid.h5'
        self.path_h5_test = path_h5_base + 'h5_' + str(self.sequence_length) + '_test.h5'
    
        # build h5 file if doesn't exists()
        if not os.path.exists(self.path_h5_train) or not os.path.exists(self.path_h5_valid) or not os.path.exists(self.path_h5_test) or not os.path.exists(path_h5_base + 'h5_meta.json'):
            
            # delete partially created cache
            paths_to_clear = [self.path_h5_train, self.path_h5_valid, self.path_h5_test, path_h5_base + 'h5_meta.json']
            for path_to_clear in paths_to_clear:                
                if os.path.exists(path_to_clear):
                    # logging.info("Removing partially created sequences cache file: {}".format(path_to_clear))
                    os.remove(path_to_clear)
             
            if verbose:
                logging.info("Computing frame sequence h5 files: {} [may take a few minutes]".format(path_h5_base))

            ##################################################
            ### get size of train/valid/test sequence datasets
            ##################################################

            # total number of rows of sequence data we have for each split
            # this is not the same as the number of frames since we exclude
            # the first (self.sequence_length-1) frames
            total_rows_train = 0
            total_rows_valid = 0
            total_rows_test = 0

            # load resized numpy array
            path_vid_resized = path_cache + 'frames/'
            path_vid_resized += str(self.frame_size[0]) + "_" + str(self.frame_size[1]) + '/' 

            path_labels = path_cache + 'labels/'

            # read vid paths
            path_videos = get_video_paths()

            # loop over all vids and resize frames, saving to new folder in /cache/frames/
            for c, path_video in enumerate(path_videos):
                                
                if verbose:
                    logging.info("Computing frame sequence h5 files: {}/{} [precompute]".format(c+1,len(path_videos)))

                # get vid name from path
                video_name = path_video.split("/")[-2]

                # load resized frames
                frames = np.load(path_vid_resized  + video_name + '.npy')

                # build sequences
                x = []
                for i in range(self.sequence_length, len(frames) + 1):
                    x.append(frames[i-self.sequence_length:i])
                x = np.array(x)

                if self.video_splits[video_name] == "train":
                    total_rows_train += len(x)
                if self.video_splits[video_name] == "valid":
                    total_rows_valid += len(x)
                if self.video_splits[video_name] == "test":
                    total_rows_test += len(x)

            # calc shapes required for full sequence dataset
            if self.sequence_length > 1:
                h5_shape_train_x = (total_rows_train, self.sequence_length, self.frame_size[0], self.frame_size[1], 3)
                h5_shape_valid_x = (total_rows_valid, self.sequence_length, self.frame_size[0], self.frame_size[1], 3)
                h5_shape_test_x = (total_rows_test, self.sequence_length, self.frame_size[0], self.frame_size[1], 3)
            else:
                h5_shape_train_x = (total_rows_train, self.frame_size[0], self.frame_size[1], 3)
                h5_shape_valid_x = (total_rows_valid, self.frame_size[0], self.frame_size[1], 3)
                h5_shape_test_x = (total_rows_test, self.frame_size[0], self.frame_size[1], 3)
                
            h5_shape_train_y = (total_rows_train, self.num_classes)
            h5_shape_valid_y = (total_rows_valid, self.num_classes)
            h5_shape_test_y = (total_rows_test, self.num_classes)


            ################################
            ### Initialize and open h5 files
            ################################

            # open h5 file to store big sequence dataset feature and label arrays
            # path_h5file = MODEL -> SEQUENCE LENGTH
            f_train = h5py.File(self.path_h5_train, 'a')
            f_valid = h5py.File(self.path_h5_valid, 'a')
            f_test = h5py.File(self.path_h5_test, 'a')

            # initialize h5 datasets
            h5_train_x = f_train.create_dataset('sequences', shape= h5_shape_train_x, dtype='uint8')
            h5_train_y = f_train.create_dataset('labels', shape= h5_shape_train_y, dtype='uint8')

            h5_valid_x = f_valid.create_dataset('sequences', shape= h5_shape_valid_x, dtype='uint8')
            h5_valid_y = f_valid.create_dataset('labels', shape= h5_shape_valid_y, dtype='uint8')

            h5_test_x = f_test.create_dataset('sequences', shape= h5_shape_test_x, dtype='uint8')
            h5_test_y = f_test.create_dataset('labels', shape= h5_shape_test_y, dtype='uint8')

            ##################################################
            ### Build h5 files for this sequence / model combo
            ##################################################

            # load resized numpy array
            path_vid_resized = path_cache + 'frames/'
            path_vid_resized += str(self.frame_size[0]) + "_" + str(self.frame_size[1]) + '/' 

            path_labels = path_cache + 'labels/'

            # read vid paths
            path_videos = get_video_paths()

            # keep track of where we are in the h5 file
            h5_cursor_train = 0
            h5_cursor_valid = 0
            h5_cursor_test = 0

            # loop over all vids and build sequences file
            for c, path_video in enumerate(path_videos):
                
                if verbose:
                    logging.info("Computing frame sequence h5 files: {}/{} [build h5 file]".format(c+1,len(path_videos)))

                # get vid name from path
                video_name = path_video.split("/")[-2]

                ### create sequence: features
                # load precomputed frames
                frames = np.load(path_vid_resized  + video_name + '.npy')
                
                # first apply preprocessing if pretrained model given
                if self.pretrained_model_name != None:
                    frames = self.preprocess_input(frames.astype(np.float32))
                    
                # build sequences
                x = []
                for i in range(self.sequence_length, len(frames) + 1):
                    x.append(frames[i-self.sequence_length:i])
                x = np.array(x)

                ### create sequence: labels
                # load precomputed labels
                labels = np.load(path_labels + video_name + '.npy')     

                # temp lists to store sequences
                y = []
                for i in range(self.sequence_length, len(labels) + 1):
                    y.append(labels[i-1])
                y = (np.array(y))

                ### write this vid's data to relevant h5 dataset
                if self.sequence_length > 1:
                    if self.video_splits[video_name] == "train":
                        h5_train_x[h5_cursor_train:h5_cursor_train + x.shape[0], :, :, :, :] = x
                        h5_train_y[h5_cursor_train:h5_cursor_train + y.shape[0], :] = y
                        h5_cursor_train += len(x)
                    if self.video_splits[video_name] == "valid":
                        h5_valid_x[h5_cursor_valid:h5_cursor_valid + x.shape[0], :, :, :, :] = x
                        h5_valid_y[h5_cursor_valid:h5_cursor_valid + y.shape[0], :] = y
                        h5_cursor_valid += len(x)
                    if self.video_splits[video_name] == "test":
                        h5_test_x[h5_cursor_test:h5_cursor_test + x.shape[0], :, :, :, :] = x
                        h5_test_y[h5_cursor_test:h5_cursor_test + y.shape[0], :] = y
                        h5_cursor_test += len(x)
                else:
                    # remove sequence_length dimension
                    x = np.squeeze(x)
                    
                    if self.video_splits[video_name] == "train":
                        h5_train_x[h5_cursor_train:h5_cursor_train + x.shape[0], :, :, :] = x
                        h5_train_y[h5_cursor_train:h5_cursor_train + y.shape[0], :] = y
                        h5_cursor_train += len(x)
                    if self.video_splits[video_name] == "valid":
                        h5_valid_x[h5_cursor_valid:h5_cursor_valid + x.shape[0], :, :, :] = x
                        h5_valid_y[h5_cursor_valid:h5_cursor_valid + y.shape[0], :] = y
                        h5_cursor_valid += len(x)
                    if self.video_splits[video_name] == "test":
                        h5_test_x[h5_cursor_test:h5_cursor_test + x.shape[0], :, :, :] = x
                        h5_test_y[h5_cursor_test:h5_cursor_test + y.shape[0], :] = y
                        h5_cursor_test += len(x)
            
            # save total row counts to file
            with open(path_h5_base + 'h5_meta.json', 'w') as fp:
                json.dump({'total_rows_train':total_rows_train,
                           'total_rows_valid': total_rows_valid,
                           'total_rows_test':total_rows_test}
                          , fp)
                    
            # close h5 files
            f_train.close()
            f_valid.close()
            f_test.close()
        
            # return total samples for each split so we can pass them to our DataGenerator
            return total_rows_train, total_rows_valid, total_rows_test
        
        else:
            # h5 sequence file already exists - just load the sequence meta and return sequence lengths 
            # so we can pass them to our DataGenerator
            total_rows_train, total_rows_valid, total_rows_test = None, None, None
            
            with open(path_h5_base + 'h5_meta.json', 'r') as fp:
                h5_meta = json.load(fp)
                total_rows_train = h5_meta['total_rows_train']
                total_rows_valid = h5_meta['total_rows_valid']
                total_rows_test = h5_meta['total_rows_test']
                
            # return total samples for each split so we can pass them to our DataGenerator
            return total_rows_train, total_rows_valid, total_rows_test


# In[28]:


### init generator
# data = Data(sequence_length = 1, 
#             return_CNN_features = False, 
#             pretrained_model_name="vgg16",
#             pooling = "max",
#             return_generator=True,
#             batch_size=32)
#
#
### View batches
# dd = []     # store all the generated data batches
# labels = []   # store all the generated label batches
# max_iter = 100  # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
# i = 0
# for d, l in data.generator_test:
#     dd.append(d)
#     labels.append(l)
#     i += 1
#     if i == max_iter:
#         break


# In[94]:


if __name__ == "__main__":
    
    logger.info("Building resized and feature cache")
    # build feature cache in advance by running python3 data.py
    for pretrained_model_name in pretrained_model_names:
        for pooling in poolings:
            logger.info("Building resized and feature cache: {}, {}".format(pretrained_model_name, pooling))
            data = Data(sequence_length=1, 
                        return_CNN_features=True,
                        pretrained_model_name = pretrained_model_name,
                        pooling=pooling)
            

    # build h5 cache
    logger.info("Building h5 cache")
    for sequence_length in [2,3,5,10,20]:
        for pretrained_model_name in pretrained_model_names:
            logger.info("Building h5 cache: {}, {}".format(sequence_length, pretrained_model_name))
            data = Data(sequence_length=sequence_length, 
                        return_CNN_features=False, 
                        pretrained_model_name = pretrained_model_name, 
                        return_generator = True,
                        verbose=True, batch_size=32)

