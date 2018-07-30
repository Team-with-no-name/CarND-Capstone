This directory contains an implementation of image based traffic light classification.

# tl_classifier.py

Implements communication between ros and traffic light models.

Sends raw images from ros to model(s) and produces a prediction of the
traffic light state represented by the image.

# Keras implementation of classifier to predict traffic light state from whole image

train_classifier.py: keras implementation of the training process for an image classification model with dataset sourced using ImageDataGenerator and flowFromDirectory.

Originally hoped to implement minimum viable product via this method, but this approach didn't seem to peform that well so started looking into more advanced models. (see below)

# Tensorflow implementation of Semantic Segmentation network to predict traffic light state

## Train the network

0.  Download the [bosch small traffic light dataset](https://hci.iwr.uni-heidelberg.de/node/6132), and the [UCSD Lisa traffic light dataset](http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/traffic-light-detection/)

1.  Concatenate the parts and unzip the bosch dataset into a directory called 'bosch_train': `cat dataset_train_rgb.zip.* > dataset_train_rgb.zip; unzip dataset_train_rgb.zip -d ./data_dir/bosch_train`
1.  Import training data from bosch dataset (produces segmented images): `python dataset.py --bosch-file ./path_to/data_dir/bosch_train/train.yaml`
1.  Convert the LISA data to bosch format and import `python dataset.py --lisa-dir ./path/to/unzipped_lisa_data/`
1.  Copy the Lisa Directory into bosch_train directory
1.  Run the training script providing path to directory containing bosch_train (trained model will be written to this directory): `python train_segmenter.py --mode train --data-dir ./path_to/data_dir --dataset traffic-lights --num-epochs 10 --num-validation-steps 10 --batch-size 10`
1.  Test images will be written each epoch to `./path_to/data_dir/trafficlight-segmenter/predictions`
1.  A model checkpoint and a frozen model file will be written each epoch to: `./path_to/data_dir/trafficlight-segmenter/model`

## Resume training the network from checkpoint (fine-tuning or continuing after interruption)

`python train_segmenter.py --mode train --data-dir ./path_to/data_dir --dataset traffic-lights --num-epochs 10 --num-validation-steps 10 --batch-size 10 --restore-from-checkpoint`

## Produce some images annotated with the network

0.  `python train_segmenter.py --mode test --dataset traffic-lights --use-frozen-model path/to/saved_model-.pb --test-images-from-directory path/to/test_images`

## Description

train-segmenter.py produces a semantic segmentation network that attempts to predict for each pixel in an image, whether that pixel is part of a 'red light' 'green light' 'yellow light' or none of those.

The network consists of a pre-trained encoding layer based on a fully convolutional version of vgg16. The encoding is then fed to a 'decoding layer' consisting of a stack of convolutional upsampling layers + skip layers. The end output of the decoding layer is an image with same dimensions as the input, where each pixel location is a class value.

There was initially trouble training this network for this task. It seemed there was a severe class imbalance issue because the vast majority of the images in the training set consisted primarily of 'unknown'/background class. The objects we are interested in (traffic lights) are fairly small portion of the images in the training set. A couple of strategies were employed to try to work around this:

1.  loss_function="cross_entropy": First attempted to train the network with cross entropy loss function. This led to a network which converged on predicting all background for every pixel in the output image due to the extreme class imbalance
2.  loss_function="weighted_cross_entropy": Next tried to apply weighting scheme to classes to see if could correct for the class imbalance -- this also didn't appear to work.
3.  loss_function="iou_estimate": Next tried using a differentiable estimate of the intersection over union metric as loss function. Still wouldn't converge.
4.  loss_function="weighted_iou_estimate": Next used the same differentiable estimate of iou as loss function -- and also applied weighting to address unbalanced classes.
5.  Added cropped versions of the training images with higher ratio of non-background to background by cropping regions around the labeled areas of interest (plus a small buffer of additional pixels). Should have gone back re-rested the other loss functions to see if they could converge on this improved dataset but didn't get to it.

The network performs pretty well after a few epochs of training. Here are some sample images produced by the network.

And here are a few images of predictions for images in the simulator (NOTE: no simulator images were included in training set )

## Runtime Performance of Model

More work is needed to optimize the runtime performance of the model -- but its unclear whether we will have time to get to it. Some investigations that should happen but might not have time ...

1.  The image size supplied to the model could be reduced (the model still works on images half the size being supplied - and maybe smaller)
1.  The model was frozen which decreases the disk storage requirements -- but this frozen graph could be significantly optimized via graph transformation for perf and memory size.
1.  It seem there must be something we could do within the ROS machinery/project code to address performance problems outside of the model ... Perhaps resizing images in bridge.py _before_ they get sent into the kludgey r.o.s machinery could help ... At the time of these notes, there is some component in the system _somewhere_ that causes images to buffer -- so when the model takes too long processing an image, the next images to arrive at the model are very old. This accumulating latency completely breaks performance of the system.

Maybe the 'correct' workaround for performance issues is to 'skip frames' prior to supplying to the model -- but not sure a robust way to tune the number of frames to skip so the number of images skipped will match the model processing time ... The model processing time is VASTLY different depending on whether the model is running on a CPU or GPU system. GPU system's capable of running this ros environment (require's native ubuntu installation because of the ... fragility ... that is the ros ecosystem) -- are hard to come by ...

## TODO: Accuracy on parking lot bag

The images in the parking lot bag are wildly oversaturated and have bizarre color ranges. At time of writing the model doesn't perform very well on these images. A small dataset of labeled data was created for these images via labelme -- and this dataset was added to the overall training set. The images were not included in the original training set for the model but rather added to it at some point along the way. Accuracy didn't seem to improve on these images after a few epochs of training with them included -- but these are an incredibly small number of images relative to the training set size. Its possible might be able to fine-tune the model and get it to work on these images by training a few epochs on a smaller dataset that includes equal numbers of image images from these sets as from the other sets -- or maybe if these images had been included in the training set initially, perhaps this issue wouldn't have developed in the first place ...

## TODO: Retraining the model from scratch

The network/training process/training datasets were modified -- and incorporated into the model by resuming from checkpoints. Because of how long it takes to train the model -- haven't gotten to retraining from scratch with the latest code/datasets.
