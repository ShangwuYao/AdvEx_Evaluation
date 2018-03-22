"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from PIL import Image
import glob
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
from sklearn.model_selection import train_test_split

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level

import os

FLAGS = flags.FLAGS


def get_imagenet(data_dir="dataset/tiny-imagenet-200/train/"):
    image_list = []
    object_classes = [name for name in os.listdir(data_dir) if not name.startswith(".")]
    image_collections = []
    running_sum_x = 0
    running_sum_y = 0
    count = 0
    labels = []
    label = 0
    for object_name in object_classes:
        cur_dir=data_dir+object_name+'/images/'
        images=[image for image in os.listdir(cur_dir)]
        for image in images:
            object_images = []
            im=Image.open(cur_dir+image)
            pix = np.array(im)

            if len(pix.shape) < 3:
                # grayscale image to RGB (3 dimension)
                pix = cv2.cvtColor(pix, cv2.COLOR_GRAY2BGR) 
            image_reshaped = cv2.resize(pix, (300, 244), interpolation = cv2.INTER_CUBIC) # reshape based on average shape
            object_images.append(image_reshaped)
            labels.append(label)

            # get the running sum for uniform shape
            count += 1
            running_sum_x += 1 / count * (pix.shape[0] - running_sum_x)
            running_sum_y += 1 / count * (pix.shape[1] - running_sum_y)
            
        
        object_images = np.concatenate([object_[np.newaxis,:,:,:] for object_ in object_images])    
        image_collections.append(object_images)
        label += 1

    print(running_sum_y, running_sum_x) # get the average shape
        
    X = np.concatenate(image_collections)    
    Y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=8080) # shuffle is default

    return X_train, X_test, y_train, y_test

get_imagenet()


