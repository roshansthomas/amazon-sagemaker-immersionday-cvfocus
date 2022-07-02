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

## Lab 2

In this lab, we will train a machine learning model using the bring your own script method. We will train a model using user-defined u-net model implemented with the MX-Net framework and train a semantic segmentation model from the torch vision model zoo using the pytorch framework.

### Setup

Before we begin, we will setup an Amazon SageMaker Notebook which will pre-install some packages required to convert the SpaceNet dataset from geojson into images using the Solaris library. This can be done by deploying the lab2.yml cloudformation template in your account.

1. Download the [lab2.yml](./lab2.yml) file
2. In your AWS account, go to [Cloudformation](https://ap-southeast-2.console.aws.amazon.com/cloudformation/home?region=ap-southeast-2#/)
3. Next click `Create Stack`
4. Upload a template, select the lab2.yml file and click next.
5. Provide the stack with a name (i.e. immersion-day-lab2)
6. Click `Next` twice and tick the checkbox to acknowledge creation of IAM resources and click `Create Stack`

Note: This setup will take at approximate 20 minutes to install the conda packages in the Amazon SageMaker notebook.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

