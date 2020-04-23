> WORK IN PROGRESS

# Deep Neural Networks for Video Classification

This repository can be used to train deep neural networks for video classification. It also contains several Jupyter notebooks to transform data into the format required and to analyze model outputs.

These models were implemented for use in Ecology but can be used in any application. They were developed with the following applications in mind:

1. Detect animals in camera trap videos.
2. Classify animal actions in video captured "first-person" animal-mounted cameras.

The following models are implemented (more information is given in the `Training Models` section): 

* Image-only Convolutional Neural Network (CNN) - input is a single frame. Baseline that ignores temporal information in the video. Best practice (implemented here) is to use "transfer learning" and fine-tune a CNN architecture that has already been trained on ImageNet. 
* Concatenated frames CNN - input is a concatenated clip of frames (`sequence length`) 

## Setup
This code is intended to be run on a machine with a GPU. It could be run locally or using a cloud provider such as Amazon Web Services or Google Cloud Platform.

The easiest way to get started is to create a virtual machine with a GPU on one of the cloud provider platforms using their deep learning image which will install and configure TensorFlow to be used with the GPU. 

You may need to `pip install` some packages listed in `requirements.txt`. Check your package versions or submit an issue if you run into any errors.

## Training Models
A single model can be trained using the `train_single_model.ipynb` notebook. Parameters are set in the experiment dictionary at the top of the notebook. 

Here is an explanation of the parameters that can be used for an experiment:

* `model_id` integer identifier for this model e.g. 1234 - model will be saved in `/models/model_id`    
* `architecture` architecture of model in [image_MLP_frozen, image_MLP_trainable, video_MLP_co video_LRCNN_frozen, video_LRCNN_trainable, C3D, C3Ds
* `sequence_length` number of frames in sequence to be returned by Data object
* `frame_size` size that frames are resized to (different models / architectures accept different input si will be inferred if pretrained_model_name is given since they have fixed sizes)
* `pretrained_model_name` name of pretrained model (or None if not using pretrained model e.g. for 3D-CNN)
* `pooling` name of pooling variant (or None if not using pretrained model e.g. for 3D-CNN or if fitting mor-dense layers on top of pretrained model 
* `sequence_model` sequence model in [LSTM, SimpleRNN, GRU, Convolution1D]
:sequence_model_layers` default to 1, can be stacked 2 or 3 (but less than 4) layer sequence model (always stacking the same sequence model, not mixing LSTM and GRU, for example)
* `layer_1_size` number of neurons in layer 1
* `layer_2_size` number of neurons in layer 2
* `layer_3_size` number of neurons in layer 3
* `dropout` amount of dropout to add (same applied throughout model - good default is 0.20)
* `convolution_kernel_size` size of 1-D convolutional kernel for 1-d conv sequence models (good default is 3)
* `model_weights_path` path to .h5 weights file to be loaded for pretrained CNN in LRCNN-train        
* `batch_size` batch size used to fit model (default to 32)
* `verbose` whether to log progress updates


## Analyzing a Trained Model
The notebook `model_analysis.ipynb` can be used to load metrics about model training including loss curve statistics and other metadata produced during model training. 

The `results.json` file located in the trained model directory contains 

```
{
    "architecture": "video_lrcnn_frozen",
    "batch_size": 32,
    "convolution_kernel_size": 3,
    "data_total_rows_test": 265,
    "data_total_rows_train": 10034,
    "data_total_rows_valid": 1285,
    "dropout": 0.2,
    "fit_best_round": 3,
    "fit_dt_test_duration_seconds": "0",
    "fit_dt_test_end": "2020-04-07 10:50:30",
    "fit_dt_test_start": "2020-04-07 10:50:29",
    "fit_dt_train_duration_seconds": "925",
    "fit_dt_train_end": "2020-04-07 10:50:28",
    "fit_dt_train_start": "2020-04-07 10:35:02",
    "fit_num_epochs": 24,
    "fit_stopped_epoch1": 12,
    "fit_stopped_epoch2": 4,
    "fit_stopped_epoch3": 5,
    "fit_test_acc": 0.7962264150943397,
    "fit_train_acc": 0.8900737492763025,
    "fit_train_loss": 0.2812534705062822,
    "fit_val_acc": 0.9097276265055289,
    "fit_val_loss": 0.252977742005415,
    "frame_size": [
        224,
        224
    ],
    "layer_1_size": 256,
    "layer_2_size": 512,
    "layer_3_size": 256,
    "model_id": 1,
    "model_param_count": 4984578,
    "model_weights_path": null,
    "num_features": 512,
    "path_model": "/mnt/seals/models/1/",
    "pooling": "max",
    "pretrained_model_name": "vgg16",
    "sequence_length": 20,
    "sequence_model": "LSTM",
    "sequence_model_layers": 2,
    "verbose": true
}
```

![Loss curve](https://raw.githubusercontent.com/alxcnwy/Deep-Neural-Networks-for-Video-Classification/master/readme/accuracy_example.png)

![Confusion Matrix](https://raw.githubusercontent.com/alxcnwy/Deep-Neural-Networks-for-Video-Classification/master/readme/confusion_example.png)

## Loading a Trained Model & Predicting Frames
The `load_model_and_predict_frames.ipynb` notebook can be used to load a trained model and use it to output predictions for each frame in the dataset. 

It produces a file in the model directory called `frame_predictions.csv` with the following columns:

* `class 1` - predicted probability for class 1
* `class 2` - predicted probability for class 2
*  `...`
*  `class n` - predicted probability for class n
*  `prediction` - class with max probability
*  `video` - video name
*  `frame` - frame filename
*  `label` - label for given frame
*  `split` - train/valid/test split
*  `error` - whether an error was made

If labels are unknown, a dummy label equal to one of the labels used by the model should be given in `labels.csv`.


## Helper Notebooks
There are several helper notebooks included in the `/notebooks/` directory.

### > `helper_extract_frames.ipynb`
This notebook can be used to 

### > `helper_convert_timestamps_file_to_labels.ipynb`
This notebook can be used to 

### > `helper_check_frames_against_labels.ipynb`
This notebook can be used to 

### > `helper_add_train_valid_test_splits_to_labels.ipynb`
This notebook can be used to 

### > `helper_explore_dataset.ipynb`
This notebook can be used to 


## Researchers
* Alex Conway (UCT, www.NumberBoost.com)
* Dr. Ian Durbach (UCT, AIMS)
