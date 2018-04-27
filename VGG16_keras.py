
import numpy as np
import tensorflow as tf
import keras
import json
import os
from keras.layers import Input
from keras import backend
from keras import utils
from cleverhans.attacks import CarliniWagnerL2
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions


path='./dataset/tiny-imagenet-200/train/n01443537/images'
CLASS_INDEX=json.load(open('./imagenet_class_index.json'))
def vgg16_evaluate():

    sess = tf.Session()
    keras.backend.set_session(sess)
    
    ##Load images for evaluation. Took Stanford 231n tiny set for testing (goldfish)
    images=[]
    target=np.ones(100)
    for index,myfile in enumerate(os.listdir(path)):
        if index==100:
            break
        if myfile.endswith('JPEG'):
            image=load_img(path+'/'+myfile,target_size=(224,224))
            inputs=img_to_array(image)
            inputs=inputs.reshape(1,inputs.shape[0],inputs.shape[1],inputs.shape[2])
            images.append(inputs)
            #target.append(np.zeros(1000))
            #target[-1][1]=1

    target=utils.to_categorical(target,1000)
    x_input=np.vstack(images)
    x_input=preprocess_input(x_input)
    model=VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#    y=model.evaluate(x_input,target,verbose=1)
#    print(y)

    cw_attack=CarliniWagnerL2(model=model,back='tf',sess=sess)
    
    ##Untargeted cw_attack parameters
    cw_params = {'binary_search_steps': 1,
                 'y_target': None,
                 'max_iterations': 10,
                 'learning_rate': 0.1,
                 'batch_size': 100,
                 'initial_const': 10}
    adv_inputs=x_input[:]
    adv=cw_attack.generate_np(adv_inputs,**cw_params)
    
    adv_y=model.evaluate(adv,target,verbose=1)
    print(adv_y)

vgg16_evaluate()

    
    
    


