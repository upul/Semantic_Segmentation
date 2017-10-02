# Semantic Segmentation using a Fully Convolutional Neural Network

### Introduction
This repository contains a set of python scripts for you to train and test semantic segmentation using a fully convolutional neural network. Our semantic segmentation network is based on the [paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) described by Jonathan Long et al.

#### How to Train the Model
1. Since the network using VGG-16 weights, first, you have to download VGG-16 pre-trained weights from [https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) and save in the the `pretrained_weights` folder.
2. Download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) and save it in the data\data_road folder.
3. Next, open a command window and type `python fcn.py` and hit the enter key.

Please note that training checkpointing will be saved to `checkpoints\kitti` folder and logs will be saved to `graphs\kitti` folder. So by using `tensorboard --logdir=graphs\kitti` command, you can start tensorboard to inspect the training process.

Following shows sample output we managed to obtain during testing time.

![img_1](./sample_output/um_000014.png)
![img_1](./sample_output/um_000032.png)
![img_1](./sample_output/uu_000022.png)
![img_1](./sample_output/uu_000099.png)

### Network Architecture

We implement the `FCN-8s` model described in the (paper)[https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf] by Jonathan Long et al. Following figure shows the architecture of the network. We generated this figure using TensorBoard.

![architecture](./images/fcn_graph.png)

### The KITTI dataset

For training the semantic segmentation network, we used the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). The dataset consists of 289 training and 290 test images. It contains three different categories of road scenes:

* uu - urban unmarked (98/100)
* um - urban marked (95/96)
* umm - urban multiple marked lanes (96/94)

### Training the Model

![loss_graph](./images/)

### Future Improvements

### Conclusiotn

