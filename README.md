*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Using Machine Learning to Predict Titanic Disaster Survival

My capstone project for Udacity's Machine Learning Engineer Nanodegree focuses on predicting which passengers survived the Titanic shipwreck, based on the dataset containing passenger data (ie name, age, gender, socio-economic class, etc). i.e. we are trying to build a predictive model that answers the question: “what sorts of people were more likely to survive?” The objective of this project falls into the category of classification. 

In order to approach this problem, I made use of both Azure's automated ML (AutoML) capabilities and Azure's HyperDrive hyperparameter tuning tool. The best models from each experiment were compared to find the most performant model and the best model was then deployed as an Azure container instance (ACI) for consumption. Both automated ML and Hyperdrive produced models with somewhat similar performance as assessed by Accuracy. AutoML model had an accuracy of 0.8249 and HyperDrive model had an accuracy of 0.7636.

The below diagram shows and overview of the workflow including the main tasks: 
![alt text](/img/capstone-diagram.png)

## Dataset

### Overview
Dataset used for this project is "Titanic - Machine Learning from Disaster" dataset from [Kaggle](https://www.kaggle.com/c/titanic). Aim here is to predicts which passengers survived the Titanic shipwreck. Dataset contains 12 columns and 891 rows of data, including both categorical and continuous data.

Dataset features:

| Variable | Definition | Key |
|----------|------------|-----|
|survival|Survival|0 = No, 1 = Yes|
|pclass|Ticket class|1 = 1st, 2 = 2nd, 3 = 3rd|
|sex|Sex| |	
|Age|Age in years| |	
|sibsp|# of siblings / spouses aboard the Titanic| |
|parch|# of parents / children aboard the Titanic| |	
|ticket|Ticket number| |	
|cabin|Cabin number| |	
|fare|Passenger fare| |	
|embarked|Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton|
### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
