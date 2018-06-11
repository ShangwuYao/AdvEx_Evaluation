# Evaluation Core Algorithm

The following example creates a pipeline from generating adversarial images from different attack methods to evaluate the accuracy from those methods.

```bash
python preprocess_adversarial_images.py        # Generate images from attack method we choose
```
The `preprocess_adversarial_images.py` script set up a white-box attack schema and uses five attacks (FGSM,I-FGSM,MI-FGSM,MADRY,CW) to generate adversarial exampels.



```bash
python evaluate.py --model your_model --label your_label
```
To run the `evaluate` script, you will have to provide your_model that is saved by keras's save function `model.save` and a JSON file that contains a mapping between index to label. More detailed informaiton can be found here. [(Help)](http://google.com)
