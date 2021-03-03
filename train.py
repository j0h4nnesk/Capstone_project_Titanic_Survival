from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset

run = Run.get_context()
ws = run.experiment.workspace

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    
    url = 'https://raw.githubusercontent.com/j0h4nnesk/Capstone_project_Titanic_Survival/main/train.csv'
    dataset = TabularDatasetFactory.from_delimited_files(url)
    
    x = dataset.to_pandas_dataframe().dropna(inplace=True)
    x = x.drop('Cabin', axis=1)
    sex = pd.get_dummies(x['Sex'],drop_first=True)
    embark = pd.get_dummies(x['Embarked'],drop_first=True)
    x = pd.concat([x,sex,embark],axis=1)
    x.drop(['Sex','Embarked','Ticket','Name'],axis=1)
    x.drop(['PassengerId'],axis=1)
    y = x.pop("Survived")
    x = x.drop("Survived",axis=1)

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
    
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)

    joblib.dump(value=model, filename='outputs/model.pkl')


if __name__ == '__main__':
    main()
