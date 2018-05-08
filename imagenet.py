"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with Keras.
It is very similar to mnist_tutorial_tf.py, which does the same
thing but without a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils import other_classes, set_log_level
from cleverhans.attacks import *
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from utils import get_caltech101, make_vgg16, get_imagenet
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras import backend as K
from attack_method_config import ATTACK_DICT
import os
import time

FLAGS = flags.FLAGS


def run_imagenet(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001, train_dir="./save_weights_imagenet",
                   filename="imagenet.ckpt", load_model=False,
                   testing=False, use_pretrained=True, nb_classes=200, source_samples=10):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param train_dir: Directory storing the saved model
    :param filename: Filename to save model under
    :param load_model: True for load, False for not load
    :param testing: if true, test error is calculated
    :return: an AccuracyReport object
    """
    img_rows = 224
    img_cols = 224
    channels = 3

    keras.layers.core.K.set_learning_phase(0)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    #X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
    #                                              train_end=train_end,
    #                                              test_start=test_start,
    #                                              test_end=test_end)
    #X_train, X_test, Y_train, Y_test = get_caltech101() 
    X_train, X_test, Y_train, Y_test = get_imagenet()

    # Use label smoothing
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / float(nb_classes), 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    model = make_vgg16(lr=learning_rate, class_num=nb_classes, use_pretrained=use_pretrained)
    model.summary()
    preds = model(x)
    #preds = K.softmax(model(x))

    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        #assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train a model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': train_dir,
        'filename': filename
    }

    # load and save
    ckpt = tf.train.get_checkpoint_state(train_dir)
    ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path

    rng = np.random.RandomState([2017, 8, 30])
    if load_model and ckpt_path:
        print("------loading model------")
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        print("Model loaded from: {}".format(ckpt_path))
        print("------evaluating------")
        #evaluate() # commented for efficiency. uncomment later
    else:
        print("------Model was not loaded, training from imagenet pretrained.------")
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, save=True, rng=rng)

    # Calculate training error
    if testing:
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_train, Y_train, args=eval_params)
        report.train_clean_train_clean_eval = acc


    start_time = time.time()

    # Instantiate a attack object
    attackmethod = None
    attack_params = None

    method_string = 'mim'
    print('-------attacking method:{}-------'.format(method_string))

    #wrap = KerasModelWrapper(model)
    attackmethod = ATTACK_DICT[method_string]['attackmethod'](model, back='tf', sess=sess)
    attack_params = ATTACK_DICT[method_string]['attack_params']

    num_eval = 100
    # accept float input
    X_test = X_test.astype('float')
    idx = 0 
    accs = 0.
    cnt = 0
    for i in range(int(num_eval / 10)):
        adv_input = preprocess_input(X_test[idx:idx+10]) # avoid OOM

        attack_params['y_target'] = Y_test[idx:idx+10]

        adv = attackmethod.generate_np(adv_input, **attack_params)

        adv_y = model.evaluate(adv, Y_test[idx:idx+10], verbose=1)
        idx += 10
        cnt += 1
        accs += adv_y[1]
        print(adv_y)

    print("accuracy: {}".format(accs / cnt))

    # Close TF session
    sess.close()

    print("--- %s seconds ---" % (time.time() - start_time))    


def main(argv=None):
    run_imagenet(nb_epochs=FLAGS.nb_epochs,
                   batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   train_dir=FLAGS.train_dir,
                   filename=FLAGS.filename,
                   load_model=FLAGS.load_model,
                   use_pretrained=FLAGS.use_pretrained)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 15, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 32, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')
    flags.DEFINE_string('train_dir', './save_weights_imagenet', 'Directory where to save model.')
    flags.DEFINE_string('filename', 'imagenet.ckpt', 'Checkpoint filename.')
    flags.DEFINE_boolean('load_model', False, 'Load saved model or train.')
    flags.DEFINE_boolean('use_pretrained', True, 'Use pretrained model from imagenet or not.')
    tf.app.run()



