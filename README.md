## ANZ-Amazon-SageMaker-ImmersionDay-workshop-CV-Focus

Welcome to Machine learning with amazon sagemaker workshop

This workshop will help customers and partners to learn about the fundamentals of machine learning on amazon SageMaker with a computer vision focus.

In this hands-on session, we'll introduce Amazon SageMaker with particular focus on
- Using Amazon SageMaker Semantic Segmentation in Built Algorithm
- Bring your own script and implement semantic segmentation using Pytorch framework.

**Who should attend:**

Data scientists, Analysts, ML engineer, Data Engineers and Developers who would like to learn about solving computer vision problems using Amazon SageMaker.
Overview of the Labs

**Hands-On:**

Access to temporary AWS accounts will be provided for you on the day: No existing account required. For the best experience you may want to use a large screen or second screen if possible, to follow the workshop and hands-on side-by-side.

**Prior Knowledge:** 
Python is used as the programming language for all the labs and participants are assumed to have familiarity with Python.

**Content of this workshop:**

Lab 1) Using SageMaker Semantic Segmentation algorithm

Lab 2) Bring your own training script to train and deploy on SageMaker

### Setup

Before we begin, it is recommended you deploy the [lab2.yml](./lab2.yml) cloudformation template. The template will deploy an Amazon SageMaker Notebook with pre-installed packages which are prerequisites for lab 2. Specifically, in lab 2, we will convert the SpaceNet dataset from geojson format into images using the [Solaris](https://solaris.readthedocs.io/en/latest/) library. The steps are as follows:

1. Download the [lab2.yml](./lab2.yml) file onto your computer.
2. In your AWS account, go to the Cloudformation service.
3. Next click `Create Stack`
4. Select `Upload a template`, select the lab2.yml file and click `next`.
5. Provide the stack with a name (i.e. immersion-day-lab2).
6. Click `Next` twice and tick the checkbox to acknowledge creation of IAM resources and click `Create Stack`

Note: This setup will take at approximate 20 minutes to install the conda packages in the Amazon SageMaker notebook, so we will begin with lab 1 first.

## Lab 2

In this lab, we will train a machine learning model using the bring your own script method. We will train a model using user-defined u-net model implemented with the MXNet framework and also train a semantic segmentation model from the torch vision model zoo using the pytorch framework.

1. To begin, go to SageMaker Notebooks and open the notebook named `sagemaker-immersion-day-cv` by clicking the JupyterLab link
2. In the file explorer view, open the directory amazon-sagemaker-immersionday-cvfocus/lab2
3. We will begin with the u-net model by opening the notebook `1. SpaceNet-Workshop.ipynb`. Run through the notebook and when you kick off the training of your u-net model, open the next notebook to learn how to train a torch vision model `2. torchvision-semantic-segmentation.ipynb` as the training will take approximately 15 minutes. 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

