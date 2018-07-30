import os

from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import cv2

# Configure whether ot use the whole image classifier (implemented with keras)
# or the semantic segmentation network (implemented with just tensorflow)
USE_KERAS_CLASSIFIER = False

if USE_KERAS_CLASSIFIER:
    # Use simple whole image classifier -- (Didn't work very well)

    import keras
    import keras.models
    import h5py

    from .train_classifier import CLASS_LABELS, COLOR_IMAGE_SIZE, BW_IMAGE_SIZE

    IMAGE_SIZE = COLOR_IMAGE_SIZE

    model_name = "light_classification/tl_classifier_00.hdf5"

    class TLClassifier(object):
        def __init__(self, class_label_to_state_as_int32):

            self.class_label_to_state_as_int32 = class_label_to_state_as_int32

            if os.path.exists(model_name):
                # check that model Keras version is same as local Keras version
                f = h5py.File(model_name, mode='r')
                model_version = f.attrs.get('keras_version')
                keras_version = str(keras.__version__).encode('utf8')

                if model_version != keras_version:
                    print('You are using Keras version ', keras_version,
                          ', but the model was built using ', model_version)

                # not sure why this is needed -- workaround for issue described here ... https://github.com/keras-team/keras/issues/6462
                self.graph = tf.get_default_graph()

                self.model = keras.models.load_model(model_name)

                # do this in advance
                self.model._make_predict_function()
            self.model = None

        def get_classification(self, image):
            """Determines the color of the traffic light in the image

            Args:
                image (cv::Mat): image containing the traffic light

            Returns:
                int: ID of traffic light color (specified in styx_msgs/TrafficLight)

            """
            with self.graph.as_default():
                if IMAGE_SIZE[2] > 1:
                    resized_image = cv2.resize(image, IMAGE_SIZE[:2])
                else:
                    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    resized_image = cv2.resize(image, IMAGE_SIZE[:2])
                predicted_label = self.model.predict_classes(
                    np.expand_dims(resized_image, axis=0),
                    batch_size=1,
                    verbose=0
                )

                class_label = CLASS_LABELS[predicted_label]
                class_int = self.class_label_to_state_as_int32[class_label]
                return class_int

else:
    IMAGE_SIZE = (256, 256)  # this can be any multiple of 32 for this model
    MODEL_NAME = "./trafficlight-segmenter/saved_model-2.pb"
    from train_segmenter import restore_model

    CLASS_LABELS = [
        "unknown",
        "red",
        "green",
        "yellow"
    ]

    from .dataset import reverse_one_hot

    class TLClassifier(object):
        def __init__(self, class_label_to_state_as_int32):

            self.class_label_to_state_as_int32 = class_label_to_state_as_int32
            self.session = tf.Session()

            if os.path.exists(MODEL_NAME):
                print("Loading model from {0}".format(MODEL_NAME))
                self.logits, self.keep_prob, self.input_image, self.predict_label_probabilities, self.predict_label_distribution = restore_model(
                    self.session,
                    "this arg isn't used -- ack python is hard to maintain", MODEL_NAME
                )

            # TODO: tweak these ...?
            self.red_light_threshold = .01
            self.green_light_threshold = .01
            self.yellow_light_threshold = .01

        def get_classification(self, image):
            """Determines the color of the traffic light in the image

            Args:
                image (cv::Mat): image containing the traffic light

            Returns:
                int: ID of traffic light color (specified in styx_msgs/TrafficLight)

            """
            resized_image = cv2.resize(image, IMAGE_SIZE[:2])
            resized_image = np.expand_dims(resized_image, axis=0)

            sess = self.session

            # Determine traffic light state using semantic segmentation
            # The semantic segmentation model computes an image with same
            # dimensions where pixel locations have class values:
            # 0=(nonlight)
            # 1(red light)
            # 2(green light)
            # 3(yellow light)
            # We then compute (in tensorflow for performance) the percentage of
            # pixels in the image for each class.

            feed_dict = {
                self.input_image: resized_image,
                self.keep_prob: 1.0
            }

            label_percentages = sess.run([self.predict_label_distribution],
                                         feed_dict=feed_dict
                                         )[0]

            # The minimal viable concept for turning the segmentation
            # map into a single class prediction for the whole image.

            # A better approach might incorporate camera parameters/knowledge of the
            # known geometry of traffic light locations to determine which traffic
            # detections were most important to pay attention to ...

            # This implementation just uses a threshold --

            # if the number of 'red light' pixels exceeds a percentage of image
            # size -- its a red light ahead -- and so on with the green and yellow lights

            nonlight_percent, red_percent, green_percent, yellow_percent = label_percentages

            if red_percent > self.red_light_threshold:
                class_label = "red"
            elif green_percent > self.green_light_threshold:
                class_label = "green"
            elif yellow_percent > self.yellow_light_threshold:
                class_label = "yellow"
            else:
                class_label = "unknown"

            # print("label_percentages: ", label_percentages)
            # print("class label: ", class_label)

            return self.class_label_to_state_as_int32[class_label]
