#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:24:55 2018

@author: mancx111
"""
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import json
import pickle
import os
import scipy.io as sio
from keras.layers import Input
from keras import backend
from keras import utils
from cleverhans.attacks import FastGradientMethod,CarliniWagnerL2,BasicIterativeMethod,MadryEtAl,MomentumIterativeMethod
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,decode_predictions

'''
We use VGG16 model as our template model for black box attacks
The input of this script will be clean images
The output of this script will be adversarial images of various attacks

Notice that we generate adversarial images which have the smallest perturbance that the attack method can find.
Therefore, this gives an optimistic evaluation of the model.

'''
#path='./tiny-imagenet-200/train/n01443537/images/'
path='.'
CLASS_INDEX=json.load(open('./class_index.json'))

sess = tf.Session()
keras.backend.set_session(sess)
def load_label(path):
    mylist=[]
    with open(path,'r') as f:
        for text in f:
            mylist.append(text.strip())
            
                
    return mylist
def preprocess_input(input_):
    input_/=127.5
    input_-=1.
    return input_


def output_file(my_dict,description):
    lbl,arr=[],[]
    for key in my_dict:
        arr.extend(my_dict[key])
        lbl.extend([key]*len(my_dict[key]))
    np.save(open('evaluate_input_'+description+'.npy','wb'),arr)
    np.save(open('evaluate_label_'+description+'.npy','wb'),lbl)   

def binary_search_epsilon(model,attack,labels,output_clean,binary_iter=1,attack_params=None,description=None):

    print('Currently attack: ',description)
    ##Currently, we are planning to only take 10 images per class to generate AE
    if not attack_params:
        attack_params={'eps':None,'clip_min':0,'clip_max':255}

    num=0
    my_dict={}
    for img in os.listdir(path):
        if not img.endswith('JPEG'):
            continue
        
        lbl=int(img.split('_')[-1].split('.')[0])-1
        class_lbl=CLASS_INDEX[labels[lbl]][0]
        if class_lbl not in my_dict:
            my_dict[class_lbl]=[]
            
        iters,min_,max_=0,0.0,1.0
        image=load_img(path+'/'+img,target_size=(224,224))
        input_=img_to_array(image)

        #input_=preprocess_input(input_)
        input_=input_.reshape(1,input_.shape[0],input_.shape[1],input_.shape[2])
        #input_=input_.reshape(1,input_.shape[0],input_.shape[1],input_.shape[2])
        predict=decode_predictions(model.predict(input_))[0][0]
        
        if output_clean:
            my_dict[class_lbl].append(np.uint8(input_))
            num+=1
            if num>=5:
                output_file(my_dict,description)
                break
            continue


        if predict[0]!=class_lbl:
            print('Wrong label. Proceed')
            continue

        
        result=None
        while iters < binary_iter:
            mid=(min_+max_)/2
            attack_params['eps']=mid
            adv=attack.generate_np(input_,**attack_params).astype('int')
            yhat=model.predict(adv)
            if decode_predictions(yhat)[0][0][0]==class_lbl:
                min_=mid
                print('Noise too small',mid)
            else:
                max_=mid
                result=adv
                print('Noise too large',mid)
            iters+=1
        
        if result is None:
            continue

        my_dict[class_lbl].append(np.uint8(result))
        num+=1
        
        print('Number of AE created: ',num)
        if num>=1:
            break
    
    output_file(my_dict,description)
    
    


def main():
    #FastGradientMethod,CarliniWagnerL2,BasicIterativeMethod,MadryEtAl,MomentumIterativeMethod
    ##Use pretrained parameters for VGG16
    model=VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    attack_fgsm=FastGradientMethod(model, sess=sess)
    attack_cw=CarliniWagnerL2(model,sess=sess)
    attack_iterative=BasicIterativeMethod(model,sess=sess)
    attack_momentum=MomentumIterativeMethod(model,sess=sess)
    attack_madry=MadryEtAl(model,sess=sess)
    attacks={'CLEAN':None,'FGSM':attack_fgsm,'CW':attack_cw,'I-FGSM':attack_iterative,'Mi-FGSM':attack_momentum,'MADRY':attack_madry}
    #Load the ground truth label of the images
    labels=load_label('ILSVRC2012_validation_ground_truth.txt')
    
    #Search for the best epsilon to use
    for key in attacks:
        binary_search_epsilon(model,attacks[key],labels,key=='CLEAN',description=key)

main()
    
    
    
    
    

    
