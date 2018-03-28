
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input
from keras import backend
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions


path='./dataset/tiny-imagenet-200/train/n01443537/images'
def vgg16_evaluate():

    sess = tf.Session()
    keras.backend.set_session(sess)
    
    
    image=load_img(path+'/n01443537_0.JPEG',target_size=(224,224))
    inputs=img_to_array(image)
    inputs = inputs.reshape((1, inputs.shape[0], inputs.shape[1], inputs.shape[2]))
    inputs=preprocess_input(inputs)
    model=VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    
    yhat=model.predict(inputs)
    label=decode_predictions(yhat)
#    label=label[0]
    print(label)


vgg16_evaluate()
    
    
    


