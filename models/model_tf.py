from cleverhans_tutorials.tutorial_models import *
import math



def AlexNet(nb_classes=102,
                   input_shape=(None, 256, 256, 3)):
  layers = [Conv2D(96, (11, 11), (4, 4), "VALID"),
            ReLU(),
            Maxpool2d((3, 3), 2),
            Conv2D(256, (5, 5), (1, 1), "SAME"),
            ReLU(),
            Maxpool2d((3, 3), 2),
            Conv2D(384, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(384, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(256, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Maxpool2d((3, 3), 2),
            Flatten(),
            Linear(4096),
            ReLU(),
            Linear(4096),
            ReLU(),
            Linear(nb_classes),
            Softmax()]


  model = MLP(layers, input_shape)
  return model    

def allConv():
  '''Dropout(0.2),
        Conv2d(3, 96, 3, padding=1),
        ReLU(),
        Conv2d(96, 96, 3, padding=1),
        ReLU(),
        Conv2d(96, 96, 3, stride=2, padding=1),
        ReLU(),

        Dropout(0.5),
        Conv2d(96, 192, 3, padding=1),
        ReLU(),
        Conv2d(192, 192, 3, padding=1),
        ReLU(),
        Conv2d(192, 192, 3, stride=2, padding=1),
        ReLU(),

        Dropout(0.5),
        Conv2d(192, 192, 3),
        ReLU(),
        Conv2d(192, 192, 1),
        ReLU(),
        Conv2d(192, 10, 1),
        ReLU(),

        AvgPool2d(6),
        Flatten()'''
  layers = [Conv2D(96, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(96, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(96, (3, 3), (2, 2), "VALID"),
            ReLU(),

            Conv2D(192, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(192, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(192, (3, 3), (2, 2), "VALID"),
            ReLU(),
            
            Conv2D(192, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(192, (1, 1), (1, 1), "SAME"),
            ReLU(),
            Conv2D(10, (1, 1), (1, 1), "VALID"),
            ReLU(),

            AvgPool2d((6, 6), )
            Softmax()]


  model = MLP(layers, input_shape)
  return model        

def VGG16(nb_classes=102,
                   input_shape=(None, 112, 112, 3)):
  layers = [Conv2D(64, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(64, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Maxpool2d((2, 2), 2),
            Conv2D(128, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(128, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Maxpool2d((2, 2), 2),
            Conv2D(256, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(256, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(256, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Maxpool2d((2, 2), 2),
            Conv2D(512, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(512, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(512, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Maxpool2d((2, 2), 2),
            Conv2D(512, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(512, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Conv2D(512, (3, 3), (1, 1), "SAME"),
            ReLU(),
            Maxpool2d((2, 2), 2),
            Flatten(),
            Linear(4096),
            ReLU(),
            Linear(4096),
            ReLU(),
            Linear(nb_classes),
            Softmax()]


  model = MLP(layers, input_shape)
  return model      

class Maxpool2d(Layer):
    def __init__(self, pool_size, strides):
      self.pool_size = pool_size
      self.strides = strides

    def set_input_shape(self, shape):
        self.input_shape = shape
        size1 = int(math.floor((shape[1] - self.pool_size[0]) / self.strides)) + 1
        size2 = int(math.floor((shape[2] - self.pool_size[1]) / self.strides)) + 1
        self.output_shape = (None, size1, size2, shape[3])

    def fprop(self, x):
        return tf.layers.max_pooling2d(inputs=x, pool_size=self.pool_size, strides=self.strides)

class AvgPool2d(Layer):
    def __init__(self, pool_size, strides, padding='valid'):
      self.pool_size = pool_size
      self.strides = strides
      self.padding = padding

    def set_input_shape(self, shape):
        self.input_shape = shape
        if self.padding == 'same':
          size1 = shape[1]
          size2 = shape[2]
        else:  
          size1 = int(math.floor((shape[1] - self.pool_size[0]) / self.strides)) + 1
          size1 = int(math.floor((shape[2] - self.pool_size[1]) / self.strides)) + 1
        self.output_shape = (None, size1, size2, shape[3])

    def fprop(self, x):
        return tf.layers.average_pooling2d(inputs=x, pool_size=self.pool_size, strides=self.strides, padding=self.padding)