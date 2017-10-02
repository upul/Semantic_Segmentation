# Semantic Segmentation using a Fully Convolutional Neural Network

### Introduction
This repository contains a set of python scripts for you to train and test semantic segmentation using a fully convolutional neural network. Our semantic segmentation network is based on the [paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) described by Jonathan Long et al.

#### How to Train the Model
1. Since the network using VGG-16 weights, first, you have to download VGG-16 pre-trained weights from [https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) and save in the the `pretrained_weights` folder.
2. Download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) and save it in the data\data_road folder.
3. Next, open a command window and type `python fcn.py` and hit the enter key.

Please note that training checkpointing will be saved to `checkpoints\kitti` folder and logs will be saved to `graphs\kitti` folder. So by using `tensorboard --logdir=graphs\kitti` command, you can start tensorboard to inspect the training process.

### Network Architecture

### The KITTI dataset

### Training the Model

### Sample Output

### Future Improvements

### Conclusiotn

