import os
import glob
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split



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


