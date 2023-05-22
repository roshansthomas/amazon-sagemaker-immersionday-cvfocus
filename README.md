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

## Setup

Support has been adeded to SageMaker Studio for Lab 2. As a result, you can choose to either do the lab in SageMaker Studio or within a SageMaker notebook.

### Option 1: SageMaker Studio (Recommended)

1. In SageMaker studio, open a terminal (File -> New -> Terminal)
2. Clone this repository `git clone https://github.com/roshansthomas/amazon-sagemaker-immersionday-cvfocus.git`

### Option 2: SageMaker Notebook

Before we begin, it is recommended you deploy the [lab2.yml](./lab2.yml) cloudformation template. The template will deploy an Amazon SageMaker Notebook with pre-installed packages which are prerequisites for lab 2. Specifically, in lab 2, we will convert the SpaceNet dataset from geojson format into images using the [Solaris](https://solaris.readthedocs.io/en/latest/) library. The steps are as follows:

1. Download the [lab2.yml](./lab2.yml) file onto your computer.
2. In your AWS account, go to the Cloudformation service.
3. Next click `Create Stack`
4. Select `Upload a template`, select the lab2.yml file and click `next`.
5. Provide the stack with a name (i.e. immersion-day-lab2).
6. Click `Next` twice and tick the checkbox to acknowledge creation of IAM resources and click `Create Stack`

Note: This setup will take at approximate 20 minutes to install the conda packages in the Amazon SageMaker notebook, so we will begin with lab 1 first.

## Lab 1

In this lab, you will learn how to train a semantic segmentation algorithm using the SageMaker built-in algorithm (https://docs.aws.amazon.com/sagemaker/latest/dg/semantic-segmentation.html).

1. To begin, in the file explorer, navigate to the `amazon-sagemaker-immersionday-cvfocus` folder.
2. Open the `Lab1` folder
3. Open the notebook - `sagemaker_semantic_segmentation.ipynb` and run through the cells.


## Lab 2

In this lab, we will train a machine learning model using the bring your own script method. We will train a model using user-defined u-net model implemented with the MXNet framework and also train a semantic segmentation model from the torch vision model zoo using the pytorch framework. For this lab, you have two options, if you're using the Amazon SageMaker studio environment, go to option 1. If you're using an Amazon SageMaker notebook and want to use the solaris library to load and process the SpaceNet dataset, go to option 2.

### Option 1: SageMaker Studio

1. To begin, in the file explorer, navigate to the `amazon-sagemaker-immersionday-cvfocus` folder.
2. Optn the `Lab2` folder
3. We will begin with the u-net model by opening the notebook `1. SpaceNet-Workshop.ipynb`. Run through the notebook and when you kick off the training of your u-net model, open the next notebook to learn how to train a torch vision model `2a. torchvision-semantic-segmentation-studio.ipyn` as the training will take approximately 15 minutes. 

### Option 2: SageMaker Notebook

1. To begin, go to SageMaker Notebooks and open the notebook named `sagemaker-immersion-day-cv` by clicking the JupyterLab link
2. In the file explorer view, open the directory amazon-sagemaker-immersionday-cvfocus/lab2
3. We will begin with the u-net model by opening the notebook `1. SpaceNet-Workshop.ipynb`. Run through the notebook and when you kick off the training of your u-net model, open the next notebook to learn how to train a torch vision model `2b. torchvision-semantic-segmentation.ipynb` as the training will take approximately 15 minutes. 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

