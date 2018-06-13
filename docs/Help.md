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

After a thorough literature review of the attack methods of the [Cleverhans library](https://github.com/tensorflow/cleverhans), the following attack methods have been chosen to check the robustness of your model.

  - FastGradientMethod
  - BasicIterativeMethod
  - MomentumIterativeMethod
  - CarliniWagnerL2
  - MadryEtAl 

A brief decription of each of the attack methods and a link to their published paper has been given below.
##### FastGradientMethod
Link: https://arxiv.org/pdf/1412.6572.pdf
This is the basic algorithm for generating adversarial examples. It is also called the FastGradientSignMethod.
##### BasicIterativeMethod
Link: https://arxiv.org/pdf/1607.02533.pdf
It is an extension of the FastGradientMethod in the sense that it is applied multiple times with small step size, and pixel values of intermediate results are clipped after each step to ensure that they are in an ε-neighbourhood of the original image.
##### CarliniWagnerL2
Link: https://arxiv.org/abs/1608.04644
It is an iterative attack that finds adversarial examples on many defenses that are robust to other attacks. 
##### MomentumIterativeMethod
Link: https://arxiv.org/pdf/1710.06081.pdf
By integrating the momentum term into the iterative process for attacks, this attack method can stabilize update directions and escape from poor local maxima during the iterations, resulting in more transferable adversarial examples.
##### MadryEtAl 
Link: https://arxiv.org/pdf/1706.06083.pdf
This performs the Projected Gradient Descent Algorithm to attack the model.

 A description of the parameters of each of these attack methods can be found [here](http://cleverhans.readthedocs.io/en/latest/source/attacks.html).

