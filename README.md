# Deep Neural Networks for Video Classification in Ecology

This repository contains code related to research into using deep neural networks for video classification in ecology.

## Researchers:
* Dr. Ian Durbach (UCT, AIMS)
* Alex Conway (UCT, NumberBoost)

## Models:

### Single Frame Classification
* [x] Fine-tuned CNNs: VGG16, Xception, ResNet50, InceptionV3, InceptionResNetV2, MobileNetV2
* [ ] Fine-tuned CNN Ensemble

### Sequence Models
* [x] MLP on concatenated CNN frames 
* [x] RNN on top of CNN encoded frames
* [x] LSTM on top of CNN encoded frames
* [ ] GRU on top of CNN encoded frames
* [ ] CNN on top of CNN encoded frames

## Usage Instructions

> To do

## Setup

The model was trained on an aws `p2.xlarge` instance with the `nvidia deep learning ami`.


## Refactor

* Sequence length calc in h5 feature gen
* Generate features on resized frames instead of raw frames
* Worker should run without a worker id for single GPU training
