from sklearn.linear_model import LogisticRegression
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

subscription_id = '2d4b3a3e-de2a-45bb-9ac0-29caf8f98da4'
resource_group = 'Capstone-project'
workspace_name = 'Capstone-project'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='titanic-survival-data')
df = dataset.to_pandas_dataframe()

df.head()

x = df.drop('Survived',axis=1)
y = df[['Survived']]

# Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

'''
data = {"train": {"X": x_train, "y": y_train},
        "test": {"X": x_test, "y": y_test}}

run = Run.get_context()
'''


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
'''
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(value=model, filename='outputs/model.pkl')
    
'''

if __name__ == '__main__':
    main()
