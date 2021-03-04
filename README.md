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
The objective of this project is to classify passengers based on did they survive the Titanic shipwreck or not, 'Survived' as either 0 = *No* or 1 = *Yes*.
### Access
First I downloaded the dataset from [Kaggle](https://www.kaggle.com/c/titanic/overview) and uploaded it to my Github, and then made the data publicly accessible via this link: https://raw.githubusercontent.com/j0h4nnesk/Capstone_project_Titanic_Survival/main/train.csv
## Automated ML
Below you can see an overview of AutoML settings and configurations:
```
# Automl settings
automl_settings = {"n_cross_validations": 2,
                    "primary_metric": 'accuracy',
                    "enable_early_stopping": True,
                    "max_concurrent_iterations": 4,
                    "experiment_timeout_minutes": 30,
                    "verbosity": logging.INFO
                    }

# Parameters for AutoMLConfig
automl_config = AutoMLConfig(compute_target = compute_target,
                            task='classification',
                            training_data=dataset,
                            label_column_name='Survived',
                            path = project_folder,
                            featurization= 'auto',
                            debug_log = "automl_errors.log",
                            enable_onnx_compatible_models=False,
                            **automl_settings
                            )
```
##### AutoML settings
*n_cross_validations=2:* How many cross validations to perform when user validation data is not specified.

*primary_metric='accuracy':* The metric that Automated Machine Learning will optimize for model selection.

*enable_early_stopping=True* Whether to enable early termination if the score is not improving in the short term. 

*max_concurrent_iterations=4:* Represents the maximum number of iterations that would be executed in parallel.

*experiment_timeout_minutes=30:* Exit criteria that is used to define how long, in minutes, the experiment should continue to run. To help avoid experiment time out failures, 30 minutes was used as the timeout value.

*verbosity=logging.INFO:* The verbosity level for writing to the log file.

##### AutoML config

*compute_target=compute_target:* The compute target to run the Automated Machine Learning experiment on.

*task='classification':* The type of task to run.

*training_data=dataset:* The training data to be used within the experiment. It should contain both training features and a label column.

*label_column_name='Survived':* The name of the label column, the target column based on which the prediction is done.

*path = project_folder:* The full path to the Azure Machine Learning project folder.

*featurization= 'auto':* 'auto' / 'off' / FeaturizationConfig Indicator for whether featurization step should be done automatically or not.

*debug_log = "automl_errors.log":* The log file to write debug information to. 

*enable_onnx_compatible_models=False:* Whether to enable or disable enforcing the ONNX-compatible models. (([ONNX](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx)) can help optimize the inference of your machine learning model. Inference, or model scoring, is the phase where the deployed model is used for prediction, most commonly on production data.)

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
Details of the AutoMl run can be viewed via the RunDetails widget: 
![alt text](/img/capstone-1.png)
The best model from the automl process was a voting ensemble model, which had an accuracy of 0.8249. Below you can see metrics , and the run ID, from the best run:
![alt text](/img/capstone-2.png)
and a screenshot from Azure ML Studio showing the best models:
![alt text](/img/capstone-3.png)


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
