import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('creditcard.csv')

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# print(legit.shape)
# print(fraud.shape)

# Statistical measures of the data
# print(legit.Amount.describe())

# print(fraud.Amount.describe())

# # Compare the values for both transactions
# print(data.groupby('Class').mean())

# Under-Sampling

# Build a sample dataset containing similar distribution of normal
# transaction and fraudulent transaction

# Number of Fraud transactions = 492

legit_sample = legit.sample(n=492)

# Concatinating two df

new_dataset = pd.concat([legit_sample, fraud],axis =0)
# print(new_dataset)

# print(new_dataset['Class'].value_counts())
# print(new_dataset.groupby('Class').mean())

# Spliting the dataset into features and targets

X = new_dataset.drop(columns='Class',axis=1)
Y = new_dataset['Class']

# Split the dataset into training data and test data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

# Train the model

# Logistic regression Model

model = LogisticRegression()

# Train the model with the training data

model.fit(X_train,Y_train)

# Performance of the model

# Accuracy score

# X_train_pred = model.predict(X_train)
# training_data_accuracy = accuracy_score(X_train_pred,Y_train)

# print(f"Accuracy score on training data : {training_data_accuracy}")


X_test_pred = model.predict(X_test)
training_data_accuracy = accuracy_score(X_test_pred,Y_test)

print(f"Accuracy score on training data : {training_data_accuracy}")