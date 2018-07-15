# Semantic Segmentation

## Udacity Self Driving Car Nanodegree Program

## Introduction

The goal of this project is to construct a fully convolutional neural network based on the VGG-16 image classifier architecture for performing semantic segmentation to identify drivable road area on images from a front-facing camera on a car, using Tensorflow.

The network uses the FCN-8 architecture described in [Fully Convolutional Networks for Semantic Segmentation, Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) and is trained and tested on the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php).

## Approach

### Architecture

A Fully Convolutional Network (FCN) was used in this project because it retains the spatial information during training. This can be really helpful when trying to identify where an object is in an image. The architecture used in this project is divided into three main parts as shown in the architecture below:

1. Encoder: Pre-trained VGG16 neural network
2. 1 x 1 convolution
3. Decoder: Transposed convolutions and skip connections

[![FCN_arch](https://github.com/AhmedElshaarany/CarND-Semantic-Segmentation/raw/master/README_images/FCN_arch.jpg)](https://github.com/AhmedElshaarany/CarND-Semantic-Segmentation/blob/master/README_images/FCN_arch.jpg)

A pre-trained VGG-16 network was converted to a fully convolutional network by converting the final fully connected layer to a 1x1 convolution and setting the depth equal to the number of desired classes (in this case, two: road and not-road). Performance is improved through the use of skip connections, performing 1x1 convolutions on previous VGG layers (in this case, layers 3 and 4) and adding them element-wise to upsampled (through transposed convolution) lower-level layers (i.e. the 1x1-convolved layer 7 is upsampled before being added to the 1x1-convolved layer 4). Each convolution and transposed convolution layer includes a kernel initializer and regularizer.

### Optimizer

Cross-entropy is used as the loss function for the network, as well as an Adam optimizer.

A regularization term has been added to the loss function, to modify the "global" loss (as in, the sum of the network loss and the regularization loss) in order to drive the optimization algorithm in desired directions. Since the optimization algorithm minimizes the global loss, my regularization term (which is high when the weights are far from zero) will push the optimization towards solutions that have weights close to zero. (Link for explanation: [Regularization loss in tensorflow](https://stackoverflow.com/questions/48443886/what-is-regularization-loss-in-tensorflow))

### Hyper-parameters

The following Hyper-parameters were used for training the FCN.

| Parameter                        | Value         |
| -------------------------------- | ------------- |
| Keep Probability                 | 0.5           |
| Batch Size                       | 2             |
| Epochs                           | 20            |
| Learning Rate                    | 0.0001 = 1e-4 |
| Normalization Standard Deviation | 0.01 = 1e-2   |
| L2 Regularization                | 0.001 = 1e-3  |

## Results

Loss per batch tends to average below 0.03 after 20 epochs.

### Samples

Below are a few before-after sample images from the output of the fully convolutional network, with the segmentation class overlaid upon the original image in green.

| Before                                                       | After                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![before1](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/um_000007.png) | ![after1](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/after_um_000007.png) |
| ![before2](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/um_000047.png) | ![after2](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/after_um_000047.png) |
| ![before3](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/uu_000014.png) | ![after3](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/after_uu_000014.png) |
| ![before4](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/uu_000073.png) | ![after4](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/after_uu_000073.png) |
| ![before5](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/umm_000060.png) | ![after5](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/after_umm_000060.png) |
| ![before6](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/um_000074.png) | ![after6](/Users/iraadit/Dropbox/Courses/Udacity_Self-Driving_Car/3_TERM_3/2_SEMANTIC_SEGMENTATION/CarND-Semantic-Segmentation/doc/after_um_000074.png) |

Performance is very good but not perfect, as sometimes sidewalks or other background can wrongly be identified as road in some images, or road can be unidentied in cases of strong shadows (brightness augmentation could help on that point, I implemented it in `helper.py` but didn't make ii run because OpenCV wasn't installed on the Udacity Advanced Deep Learning Amazon AMI).





---

# ORIGINAL README

## Semantic Segmentation

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
