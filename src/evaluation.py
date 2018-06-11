#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:09:17 2018

@author: mancx111
"""
import numpy as np
import tensorflow as tf
import keras
import json
import pickle
import os
import scipy.io as sio
import argparse
from keras.models import load_model
from keras.layers import Input
from keras import backend
from keras import utils
from cleverhans.attacks import FastGradientMethod
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input


parser = argparse.ArgumentParser()
parser.add_argument("--model",help="model path")
parser.add_argument("--index",help="index path")



args = parser.parse_args()



'''
The class takes model's path and its json file's path as input
'''
class Model_Evaluator(object):
    def __init__(self,model_path,json_path):
        super(Model_Evaluator,self).__init__()
        set_path=['CLEAN','FGSM','Mi-FGSM','I-FGSM','MADRY','CW']
        
        
        ##These methods to change when integrated
        self.input_set=load_set(set_path,'./evaluate_input_')
        self.label_set=load_set(set_path,'./evaluate_label_')
        
        #self.model=load_model(model_path)
        
        #Just for alpha testing
        self.model=VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        self.model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        
        self.class_index=json.load(open(json_path))
        
    #Private functions that only be called by init
    def load_set(self,sets,path):
        for set_ in sets:
            
            
    
    #Private functions that only will be called by evaluate
    def decode_predictions(self,predict):
        argmax=np.argmax(predict,axis=1)
        
        decode=[]
        for label in argmax:
            decode.append(self.class_index[str(label)][0])
        return np.array(decode)
        
    def evaluate(self):
        '''
        The output of this model should be accuracy
        '''
        def calculate_acc(data):
            right,total=0,0
            for key in data:
                if len(data[key]) ==0:
                    continue
                y=key
    
                batch=np.vstack(data[y])
                predict=self.decode_predictions(self.model.predict(batch,batch_size=5))
                right+=len(predict[predict==y])
                total+=len(data[y])
            return float(right)/total
            #print clean_predict.shape
        
        clean=calculate_acc(self.clean_set)
        print("Clean ACC: ",clean)
        
        adv=calculate_acc(self.adv_set)
        print("ADV ACC: ",adv)        
        
            
                


model=Model_Evaluator(args.model,args.index)
model.evaluate()
        
    