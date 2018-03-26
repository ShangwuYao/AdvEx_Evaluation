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
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans_tutorials.tutorial_models import *
from cleverhans.utils import AccuracyReport, set_log_level
from model import AlexNet

import os

FLAGS = flags.FLAGS


def get_caltech101(data_dir="../101_ObjectCategories/"): # ../../../
    image_list = []
    object_classes = [name for name in os.listdir(data_dir) if not name.startswith(".")]
    image_collections = []
    running_sum_x = 0
    running_sum_y = 0
    count = 0
    labels = []
    label = 0
    for object_name in object_classes:
        object_images = []
        for filename in glob.glob(data_dir + '{}/*.jpg'.format(object_name)): 
            im=Image.open(filename)
            pix = np.array(im)
            if len(pix.shape) < 3:
                # grayscale image to RGB (3 dimension)
                pix = cv2.cvtColor(pix, cv2.COLOR_GRAY2BGR) 
            image_reshaped = cv2.resize(pix, (256, 256), interpolation = cv2.INTER_CUBIC) # reshape based on average shape
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
    onehot = np.zeros((Y.shape[0], 102)) # 102 classes
    onehot[np.arange(Y.shape[0]), Y] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, onehot, test_size=0.20, random_state=8080) # shuffle is default

    return X_train, X_test, y_train, y_test


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get MNIST test data
    #X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
    #                                              train_end=train_end,
    #                                              test_start=test_start,
    #                                              test_end=test_end)

    X_train, X_test, Y_train, Y_test = get_caltech101() 

    # Use label smoothing
    assert Y_train.shape[1] == 102
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 256, 256, 3))
    y = tf.placeholder(tf.float32, shape=(None, 102))

    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    mim_params = {'eps': 0.3, 'eps_iter': 0.06, 'nb_iter': 10, 
                     'ord': np.inf, 'decay_factor': 1.0,
                     'clip_min': 0., 'clip_max': 1.}
    bim_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.,
                  'nb_iter': 50,
                  'eps_iter': .01}   
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}              
    rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        model = AlexNet()
        preds = model.get_probs(x)

        saver = tf.train.Saver()
        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            #assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)
            save_path = saver.save(sess, "./model_acc{}.ckpt".format(acc))
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, rng=rng)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        mim = MomentumIterativeMethod(model, sess=sess)
        adv_x = mim.generate(x, **mim_params)

        #bim = BasicIterativeMethod(model)
        #adv_x = bim.generate(x, **bim_params)
        #fgsm = FastGradientMethod(model, sess=sess)
        #adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_probs(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc

        print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = AlexNet()
    preds_2 = model_2(x)
    mim2 = MomentumIterativeMethod(model_2, sess=sess)
    adv_x_2 = mim2.generate(x, **mim_params)
    # TODO
    #fgsm2 = FastGradientMethod(model_2, sess=sess)
    #adv_x_2 = fgsm2.generate(x, **fgsm_params)
    #bim2 = BasicIterativeMethod(model_2, sess=sess)
    #adv_x_2 = bim.generate(x, **bim_params)
    if not backprop_through_attack:
        # For the fgsm attack used in this tutorial, the attack has zero
        # gradient so enabling this flag does not change the gradient.
        # For some other attacks, enabling this flag increases the cost of
        # training, but gives the defender the ability to anticipate how
        # the atacker will change their strategy in response to updates to
        # the defender's parameters.
        adv_x_2 = tf.stop_gradient(adv_x_2)
    preds_2_adv = model_2(adv_x_2)

    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy

    # Perform and evaluate adversarial training
    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, rng=rng)

    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_train, Y_train,
                              args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, preds_2_adv, X_train,
                              Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracy

    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 20, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 64, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('load_pretrained', False, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
