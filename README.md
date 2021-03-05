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
As a future improvement I would run AutoML for a longer period in hopes that a more performant model would arise. If not, I would investigate further the factors influencing model decisions in an effort to better understand when the model is correctly predicting survival (for what kind of passengers) and instances where the model breaks down.

## Hyperparameter Tuning
I chose to use a custom Scikit-learn Logistic Regression model whose hyperparameters were optimized using HyperDrive. Logistic regression is best suited for binary classification models like this one. This is the main reason I chose it.

#### Parameter sampler
**RandomParameterSampling**, where hyperparameter values are randomly selected from the defined search space, was used as a sampler. It is a good choice as it is [more efficient, though less exhaustive compared to Grid search](https://www.sciencedirect.com/science/article/pii/S1674862X19300047) to search over the search space. Parameter sampler was specified using the following parameters: 

*C:* Inverse of regularization strength; must be a positive float, where smaller values specify stronger regularization.

*max_iter:* Maximum number of iterations taken for the solvers to converge.

Parameter sampler was specified as shown below:
```
# Specify parameter sampler
ps = RandomParameterSampling({
    '--max_iter' : choice(20,40,80,100,150,200),
    '--C' : choice(0.001,0.01,0.1, 0.5, 1,1.5,10,20,50,100)
}) 
```

### Results
Completion of the HyperDrive run (RunDetails widget):
![alt text](/img/capstone-4.png)

The best model obtained from Hyperdrive had an accuracy of 0.7636, which was worse than the accuracy of AutoML model.
![alt text](/img/capstone-5.png)

For further improvement I would test also other algorithms in addition to logistic regression, and I would like to also experiment with tweaking other hyperparameters and try to get better results in that. However when looking at Kaggle the competition results are mostly quite on par with the results that I have gotten here (mostly between 70-80%) so in order to get higher accuracy levels we would probably need more training/test data (now the dataset contained "only" 891 rows i.e. passenger data points), and some more feature engineering.

## Model Deployment
As the better performing model came from AutoML experiment, the best AutoML model was the one I deployed. The model was deployed as an Azure Container Instance (ACI). To query the model data is serialized to JSON and sent to the model's endpoint as an http request. For an example of code used to interact with the deployed model below is a snippet from the 'Model Deployment' section of the automl notebook (automl.ipynb).
![alt text](/img/capstone-6.png)

Where sample data passed to the model:
![alt text](/img/capstone-7.png)

As a response, the model will send back a list of predictions. In this case they will be '1' for survived or '0' for did not survive. 
![alt text](/img/capstone-8.png)

## Screen Recording
The screen recording can be found [here](https://www.icloud.com/iclouddrive/0YPSTFT5SbDY0W0wtNfASO66Q#Capstone-project) and it shows the project in action, and more specifically demonstrates:
- A working model
- Demo of the deployed model
- Demo of a sample request sent to the endpoint and its response

