# Evaluation Core Algorithm

This project provides a pipeline for generating adversarial images using different attack methods. The images can then be used to evaluate the robustness of machine learning models. We use the attacks methods from [CleverHans library](https://github.com/tensorflow/cleverhans#setting-up-cleverhans) and perform binary search to generate images with the smallest peturbation needed to form an adversarial example. More details can be found at [https://advex.org](https://advex.org).

This project is intended for users who already have some background in machine learning and understand
the basics of adversarial examples. For users that are not familiar in this field,
below is some useful resources to learn about the topic:

- [OpenAI blog](https://blog.openai.com/adversarial-example-research/)
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)

## Usage

Running the script requires the following inputs.

### Model file

Obtained using [`model.save()` method](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) in Keras.

### Index to label mapping of your model

We have provided an example in [`imagenet_class_index.json`](https://github.com/ShangwuYao/AdvEx_Evaluation/blob/master/src/imagenet_class_index.json). Unlike the original JSON file provided by ImageNet, the value should only be a string that represents the classID instead of a list.

### Data

We use [ILSVRC](http://www.image-net.org/challenges/LSVRC/2012/index) validation data to generate the images.

Just like how ImageNet validation data is organized, images are named with index, i.e. `ILSVRC2012_val_00025012.JPEG`. Here we assume that the index `25012` is the last part of the name. On the other hand, a text file stores the labels of each image starting from index 0.

Note that if the index to label mapping of the data is different from that of the model, you need to provide the mapping of index to label of the data as well. In our case, the mapping of the model is different from that of the data. For example, the label for index `96` is `toucan` in the model but `chimpanzee` in the data.

### Config file

Config file specifies the attack methods to use and their parameters. We have provided an example in the [`config.json`](https://github.com/ShangwuYao/AdvEx_Evaluation/blob/master/src/config.json).

## Example

You can see a list of all the hyperparmeters with:

```bash
python prepare_adversarial_images.py -h
```

One example of running the script is shown below.

```bash
python prepare_adversarial_images.py --model ./vgg16.h5 --class_index ./imagenet_class_index.json --num_step 1 --num_generate 10 --data_input . --data_label ILSVRC2012_validation_ground_truth.txt --data_mapping ./class_index.json --config config.json --output_original --output_path ./image_final/
```

We use three attack methods (FGSM, I-FGSM, MI-FGSM) to generate adversarial exampels.

## Evaluation

We implemented a evaluation module in [`evaluation.py`](https://github.com/ShangwuYao/AdvEx_Evaluation/blob/master/src/evaluation.py). You need to provide your model that is saved by Keras's `model.save()` function, a JSON file that contains the mapping of index to label, and the path to where the adversarial images are stored.

You can also change which attack methods you want to evaluate on by simply altering the list inside the init function

```
self.set_path = ['Original', 'FGSM', 'I-FGSM', 'Mi-FGSM']
```
