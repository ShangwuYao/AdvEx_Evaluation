# AdvEx Evaluation Worker
Evaluation tool for deep learning models in computer vision tasks, assess models' robustness against various attack methods.

Data processing pipeline based on [CleverHans](https://github.com/tensorflow/cleverhans/tree/master/cleverhans).

Supported framework by now: Keras

Supported dataset by now: ImageNet, Caltech101, MNIST

# Dependencies
- boto3 == 1.7.32
- numpy == 1.14.2
- tensorflow == 1.8.0
- keras == 2.2.0
- flask == 1.0.2
- flask-sqlalchemy == 2.3.2

# Docker image
Docker available at [docker hub](https://hub.docker.com/r/awp135/advex/tags/).
For pulling docker:
```bash
docker pull awp135/advex:evaluation
```
