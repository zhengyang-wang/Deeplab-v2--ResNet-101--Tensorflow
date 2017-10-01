# Deeplab v2 ResNet for Semantic Image Segmentation

This is an (re-)implementation of [DeepLab v2 (ResNet-101)](http://liangchiehchen.com/projects/DeepLabv2_resnet.html) in TensorFlow for semantic image segmentation on the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/). We refer to [DrSleep's implementation](https://github.com/DrSleep/tensorflow-deeplab-resnet) (Many thanks!). We do not use tf-to-caffe packages like kaffe so you only need TensorFlow 1.3.0+ to run this code.

The pre-trained ResNet-101 ckpt files are provided by DrSleep -- [here](https://drive.google.com/drive/folders/0B_rootXHuswsZ0E4Mjh1ZU5xZVU). Thanks again!

This repository serves as the second part in the tutorial project for new students in our DIVE lab.

Created by [Zhengyang Wang](http://www.eecs.wsu.edu/~zwang6/) and [Shuiwang Ji](http://www.eecs.wsu.edu/~sji/) at Washington State University.

## Introduction

The project aims to help you learn something about semantic segmentation and be ready for a related project. As an advanced project, it includes everything that may be used in segmentation tasks, like data augmentation, transfer learning, dilated/atrous convolutions, etc.

We assume that you are able to study the task by reading materials, download the data, run the code and make some modifications as requested (we will share some ideas worth to try and you will implement it). This project may result in your first paper!

It is based on a real-world dataset named PASCAL VOC 2012, which includes images with labels for each pixel. The number of classes is 21. It includes 1,464 training images, 1,449 validation images and 1,456 testing images. An augmented version has 10582 labeled raining images, which are used in this project (try to find it by yourself). Again, the testing labels are not available (you know how to handle it!).

You will use Deeplab v2 ResNet model proposed by this [paper](https://arxiv.org/abs/1606.00915) as your baseline. This repository includes the tensorflow code.

You are supposed to complete all the requirements and do a presentation by Oct 15. In the presentation, you should demonstrate your progress and will be questioned about your understanding by others.

During the project, you should at least read the following papers.

-[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
-[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
-[Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
-[DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)

You are welcome to talk to Zhengyang to get more papers worth to read or recommend good papers found by yourself.

It is not an easy project. Feel free to ask any of us for help. Good luck!

## Project requirements

1. Download the code. Read network.py and try to understand every detail. You may find it helpful to first read papers mentioned above.

- Download this repository to your own folder on the server. If you are familiar with Github, you can use the git commands. But this is not mandatory.

2. Download the augmented PASCAL VOC 2012 dataset. The original train/val/test splits are provided in this repository under the dataset folder. However, you should perform appropriate new splits as we do not have testing labels.

- To develop a good habit, organize all your folders with appropriate names.

- This time, we do not use h5 files. We use a new way as introduced in utils/image_reader.py. Try to understand it.

3. Run the code.

- Before running, read all the code and make sure that you understand it thoroughly. You will be asked questions about every detail in your presentation. This step is much harder than the first project.

- Read the next section for how to configure the network. The default setting is provided by Deeplab.

- Run the code with pre-trained ResNet-101 model (provided above), report your results. Do not tune the hyperparameters for this step because it serves as a baseline which is the reimplementation of Deeplab's model. Use tensorboard for development. 

4. (Hard part) Make your explorations.

- Talk to Zhengyang to get some ideas and implement them.

- OR generate your own ideas. But before implementing them, discuss them with others.

- Carefully design your experiments. No mistake is tolerated in your experiments.

5. Presentation.

- Show how you do every step.

- Report your results.

- Q&A.

- If things work well, let's have your first PAPER.

## System requirement

#### Programming language
Python 3.5

#### Python Packages
tensorflow-gpu 1.3.0

## Configure the network

All network hyperparameters are configured in main.py.

#### Training

num_steps: how many iterations to train

save_interval: how many steps to save the model

random_seed: random seed for tensorflow

weight_decay: l2 regularization parameter

learning_rate: initial learning rate

power: parameter for poly learning rate

momentum: momentum

is_training: whether to updates the running means and variances of batch normalization during the training

pretrain_file: the initial pre-trained model file for transfer learning

data_list: training data list file

#### Testing/Validation

valid_step: checkpoint number for testing/validation

valid_num_steps: = number of testing/validation samples

valid_data_list: testing/validation data list file

#### Data

data_dir: data directory

batch_size: training batch size

input height: height of input image

input width: width of input image

num_classes: number of classes

ignore_label: label pixel value that should be ignored

random_scale: whether to perform random scaling data-augmentation

random_mirror: whether to perform random left-right flipping data-augmentation

#### Log

modeldir: where to store saved models

logfile: where to store training log

logdir: where to store log for tensorboard

## Training and Testing

#### Start training

After configuring the network, we can start to train. Run
```
python main.py
```
The training of Deeplab v2 ResNet will start.

#### Training process visualization

We employ tensorboard for visualization.

```
tensorboard --logdir=log --port=6006
```

You may visualize the graph of the model and (training images + groud truth labels + predicted labels).

To visualize the training loss curve, write your own script to make use of the training log.

#### Testing and prediction

Select a checkpoint to test/validate your model in terms of pixel accuracy and mean IoU.

Fill the valid_step in main.py with the checkpoint you want to test. Change valid_num_steps and valid_data_list accordingly. Run

```
python main.py --option=test
```

The final output includes pixel accuracy and mean IoU.
