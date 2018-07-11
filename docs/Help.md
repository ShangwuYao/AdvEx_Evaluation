# What You Need to Upload

You need to upload 2 files:
1. Your Image Classification model (trained on Keras) to be evaluated.
    - You will need to upload the entire model, including its weight and                architecture.
    - The best way to do it is to use model.save fucntion. Here is the [reference](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).                       
2.  A JSON file that has a mapping of your classes to the ImageNet classes.

The dataset that we’ll be using for evaluating your model is the ImageNet dataset. This leads to several restrictions about the models our website can accept.

- We can only evaluate models that are trained using ImageNet dataset.
- We will need additional details about your model. Given below is a description of what you need to provide us with.

Each of the classes in ImageNet have a unique class ID. For example, n01443537 is the ID of goldfish. Thus, along with your model, you need to also upload a JSON file that contains the mapping between your output index and the ImageNet unique class ID. An example of this will be:
```sh
{"0": "n01440764",
"1": "n01443537",
"2":  "n01484850",
"3":  "n01491361",
"4":  "n01494475",
"5":  "n01496331" }
```
The key here should be the class index and the value should be the ImageNet unique class ID.


# Attacks Performed On Your Model

We will conduct a black-box attack on your model. In other words, all the images are pregenerated, and we use [VGG16](https://arxiv.org/abs/1409.1556) to generate all the adversarial images. Using block-bax attacks as well as minumum pertubation images mentioned below, we esstianlly provide the most optimistic estimation of your model.

We have used attack methods implemented by an open sourced adversarial example library, [Cleverhans](https://github.com/tensorflow/cleverhans). After a thorough literature review of the attack methods in this library, the following attack methods have been chosen to check the robustness of your model.

  - FastGradientMethod
  - BasicIterativeMethod
  - MomentumIterativeMethod

A brief decription of each of the attack methods we have used and a link to a paper describing details of that attack method have been given below.

# Inputs to your models

It makes no sense to choose the images that are misclassified by the model in the first place. Therefore, we randomly choose 1000 images that belong to the 1000 categories from the ILSVRC 2012 validation set, which are all correctly classified by the model.

One important thing to mention is that we are finding the minumum pertubation for each of the image that can be misclassified by the model. To achieve this effect, we first set the noise level (pertubation) of a image to be the middle of range 0.0 and 1.0. If the image is misclassified, we shrink the upper bound, otherwise, we raise the lower bound so on and so forth. The number of binary search iterations is set to 10 for every image.

All the images are preprocessed before feeding to the network. Here we simply minus the images by 127.5 pixel value, which results in the range from -127.5 to 127.5 for every image. It is therefore important to note that the accuracy might vary depends on how different you preprocess the images when training your model.

##### FastGradientMethod
Link: https://arxiv.org/pdf/1412.6572.pdf
This is the basic algorithm for generating adversarial examples. It is also called the FastGradientSignMethod.
##### BasicIterativeMethod
Link: https://arxiv.org/pdf/1607.02533.pdf
It is an extension of the FastGradientMethod in the sense that it is applied multiple times with small step size, and pixel values of intermediate results are clipped after each step to ensure that they are in an ε-neighbourhood of the original image.
##### MomentumIterativeMethod
Link: https://arxiv.org/pdf/1710.06081.pdf
By integrating the momentum term into the iterative process for attacks, this attack method can stabilize update directions and escape from poor local maxima during the iterations, resulting in more transferable adversarial examples.

 A description of the parameters of each of these attack methods can be found [here](http://cleverhans.readthedocs.io/en/latest/source/attacks.html).
