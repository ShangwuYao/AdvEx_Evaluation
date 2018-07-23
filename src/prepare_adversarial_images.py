#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:24:55 2018

@author: mancx111
"""
import numpy as np
import tensorflow as tf
import keras
import json
import argparse
import os
import cleverhans.attacks as cleverhans_attacks
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model

'''
The input of this script will be clean images
The output of this script will be adversarial images of various attacks

Notice that we generate adversarial images which have the smallest perturbance that the attack method can find.
Therefore, this gives an optimistic evaluation of the model.

'''

def load_label(path):
    mylist = []
    with open(path, 'r') as f:
        for text in f:
            mylist.append(text.strip())
            
    return mylist

def output_file(my_dict, description,output_path):
    lbl, arr = [], []
    for key in my_dict:
        arr.extend(my_dict[key])
        lbl.extend([key]*len(my_dict[key]))
    np.save(open(output_path+'evaluate_input_'+description+'.npy', 'wb'), arr)
    np.save(open(output_path+'evaluate_label_'+description+'.npy', 'wb'), lbl)   
    
def preprocess_image(path):
    image = load_img(path, target_size=(224,224))
    input_ = img_to_array(image)-127
    return input_.reshape(1, input_.shape[0], input_.shape[1], input_.shape[2])

def decode_predictions(class_index,predict):
    return class_index[str(np.argmax(predict))]


def binary_search_epsilon(model, class_index, attack, input_path, labels, labels_index, output_clean=False,
                          binary_iter=1, attack_params=None, num_generate=1):

    if not attack_params:
        attack_params={'eps':None, 'clip_min':-127, 'clip_max':128}

    num = 0
    my_dict = {}
    for img in os.listdir(input_path):
        if not img.endswith('JPEG'):
            continue
        
        lbl = int(img.split('_')[-1].split('.')[0])-1
        class_lbl = labels_index[labels[lbl]]  
        if class_lbl not in my_dict:
            my_dict[class_lbl] = []
            

        input_ = preprocess_image(input_path+'/'+img)
        
        predict = decode_predictions(class_index, model.predict(input_))
        
        if predict != class_lbl:
            print('Wrong classification, continue')
            continue

        
        result = None
        iters, min_, max_ = 0, 0.0, 1.0
        while iters < binary_iter and not output_clean:
            mid = (min_+max_)/2
            attack_params['eps'] = mid
            adv = attack.generate_np(input_, **attack_params).astype('int')
            yhat = model.predict(adv)
            if decode_predictions(class_index, yhat) == class_lbl:
                min_ = mid
                print('noise too small')
            else:
                print('noise too large')
                max_ = mid
                result = adv
            iters += 1
        
        if result is None and not output_clean:
            continue
        elif output_clean:
            result = input_
        
        result += 127
        my_dict[class_lbl].append(np.uint8(result))
        num += 1
        
        print('Number of AE created: ',num)
        if num >= num_generate:
            break
    return my_dict
    
    

if __name__ == '__main__':
    sess = tf.Session()
    keras.backend.set_session(sess)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Provide the path to the model", default='./vgg16.h5')
    parser.add_argument("--class_index", help="Provide index-label mapping of the model", default='./imagenet_class_index.json')
    
    parser.add_argument("--num_step", type=int, help="number of binary search step perform to search for noise", default=5)
    parser.add_argument("--num_generate", type=int, help="number of adversarail images to generate", default=100)

    parser.add_argument("--data_input", help="path to the directory that contains the image data", default='./image_data/')
    parser.add_argument("--data_label", help="path to the file that contains label", default='./ILSVRC2012_validation_ground_truth.txt')
    parser.add_argument("--data_mapping", help="the mapping of index to label of the data", default='./imagenet_class_index.json')

    parser.add_argument("--config", help="config file that contains attack method and parameters", default='./config.json')
    parser.add_argument("--output_original", action='store_true', help="whether to output the orginal images that is clasified correctly")
    parser.add_argument("--output_path", help="path to where to store the output", default='./image_final/')

    
    args = parser.parse_args()
    
    model = load_model(args.model)
    class_index = json.load(open(args.class_index))
    
    config = json.load(open(args.config))
    data_input_path = args.data_input #To save memory, We only load images when we are processing them.
    data_label = load_label(args.data_label)
    data_mapping = json.load(open(args.data_mapping))
    
    
    
    if args.output_original:
        result_dict = binary_search_epsilon(model, class_index, None, data_input_path, data_label, data_mapping, 
                                            output_clean=True, num_generate=args.num_generate)
        output_file(result_dict, 'Original', args.output_path)
    #Config file should be a dictionary with key being the attack method (exact name) and value
    #being a dictionary contains field like alias and params(params is required)
    for attack in config:
        if not hasattr(cleverhans_attacks, attack):
            print('Cleverhans does not implement '+attack+'. Or check if your name matches the function in the library')
            continue
        
        attack_method = getattr(cleverhans_attacks, attack)(model, sess=sess)
        print('Currently running: '+attack)
        #Return a dictionary with key being the classLabel and value being a list of np array(images)

        result_dict = binary_search_epsilon(model, class_index, attack_method, 
                                            data_input_path, data_label, data_mapping, output_clean=False, 
                                            binary_iter=args.num_step,
                                            attack_params = config[attack]['attack_params'], 
                                            num_generate=args.num_generate)
        
        description=(attack if 'alias' not in config[attack] else config[attack]['alias'])
        output_file(result_dict, description, args.output_path)
        
        
        
    
    
    
    
    
    
    
    
    

    
