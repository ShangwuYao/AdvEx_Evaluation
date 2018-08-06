# AdvEx Evaluation Worker
### About AdvEx
AdvEx is a web service for assessing the robustness of machine learning models with adversarial machine learning. It is designed to satisfy the following quality attributes:
- **Scalability**: achieved by auto-scaling and load-balancing using elastic beanstalk
- **Availability**: achieved by having different servers in different availability zones
- **Performance**: achieved by using GPU in evaluation workers, and having users upload their models directly to S3 buckets without going through servers first
- **Security**: achieved by using AWS security group
- **Usability**: achieved by creating helpful instructions and tutorials for users who are not machine learning experts
- **Configurability**: achieved by using Elastic Beanstalk, Docker and config files

Links:

[Project video demo](https://www.youtube.com/watch?v=KJ1zZsia5yQ) | [Front-end static demo](https://dnc1994.com/AdvEx-FE/) | [Front-end repo](https://github.com/dnc1994/AdvEx-FE) | [Back-end repo](https://github.com/ShangwuYao/AdvEx_BE) | [Evaluation worker repo](https://github.com/ShangwuYao/AdvEx_Evaluation)

### Cloud-based system architecture 

<p align="center">
<img src="https://pic-markdown.s3.amazonaws.com/region=us-west-2&tab=overview/2018-08-06-013104.png" width=600 height=500/>
</p>

### Tech Stack
<p align="center">
<img src="https://pic-markdown.s3.amazonaws.com/region=us-west-2&tab=overview/2018-08-06-021058.png" width=600 height=300/>
</p>

### Evaluation Worker Feature
- Uses a data processing pipeline to generate adversarial images using different attack methods (based on [CleverHans](https://github.com/tensorflow/cleverhans/tree/master/cleverhans))
- Evaluates deep learning models in computer vision tasks, assesses models' robustness with two evaluation metrics: robustness score and confidence, visualize results with graphs
- Handles auto-scaling and load-balancing with Elastic Beanstalk and Docker, reduces deployment time of a new version to 5 min

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
