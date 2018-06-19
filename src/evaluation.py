#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:09:17 2018

@author: mancx111
"""
import numpy as np
import time
import json
import argparse
from keras.applications.vgg16 import VGG16


parser = argparse.ArgumentParser()
parser.add_argument("--model",help="model path",default='.')
parser.add_argument("--index",help="index path",default='./imagenet_class_index.json')



args = parser.parse_args()



'''
The class takes model's path and its json file's path as input
'''
class Model_Evaluator(object):
    def __init__(self,model_path,json_path):
        super(Model_Evaluator,self).__init__()
        self.set_path=['CLEAN','FGSM','Mi-FGSM','I-FGSM']
        
        
        ##These methods to change when integrated
#        self.input_set=load_set(set_path,'./evaluate_input_')
        
        #self.model=load_model(model_path)
        
        #Just for alpha testing
        self.model=VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        self.model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        
        self.class_index=json.load(open(json_path))
        self.class_set=set([self.class_index[x][0] for x in self.class_index])  ##need change after deployment
    #Private functions that only be called by init
    def load_set(self,set_):
        my_dict={}
        inputs=np.load(open('./image_data/evaluate_input_'+set_+'.npy'))
        labels=np.load(open('./image_data/evaluate_label_'+set_+'.npy'))
        for key,input_ in zip(labels,inputs):
            if key not in my_dict:
                my_dict[key]=[]
            my_dict[key].append(input_)
        return my_dict
            
            
    
    #Private functions that only will be called by evaluate
    def decode_predictions(self,predict):
        argmax=np.argmax(predict,axis=1)
        confidence=np.max(predict,axis=1).sum()
        
        decode=[]
        for label in argmax:
            decode.append(self.class_index[str(label)][0]) #need change after deployment
        return np.array(decode),confidence
        
    def evaluate(self):
        '''
        The output of this model should be accuracy
        '''
        def calculate_acc(data):
            right,total=0,0
            for key in data:
                if len(data[key]) ==0 or key not in self.class_set:
                    continue
                y=key
    
                batch=np.vstack(data[y])
                predict,confidence=self.decode_predictions(self.model.predict(batch,batch_size=10))
                right+=len(predict[predict==y])
                total+=len(data[y])
            return float(right)/total,confidence/total
            #print clean_predict.shape
        
        
        for path in self.set_path:
            start=time.time()
            inputs=self.load_set(path)
            acc,confidence=calculate_acc(inputs)
            end=time.time()
            print(path+" ACC: ",acc)
            print(path+" Average Confidence: ",confidence)
            print(path+" TIME: ",end-start)
#        
#        adv=calculate_acc(self.adv_set)
#        print("ADV ACC: ",adv)        
        
            
                
###Your work
##specify what is the input here
model=Model_Evaluator(args.model,args.index)
model.evaluate()

##What is the output of the evaluate
        
    