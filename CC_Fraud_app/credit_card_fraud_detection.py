import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
# from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.svm import SVC
import plotly.graph_objects as go
from random import uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Using Different Models For CCD Fraud Detection
class CCD_Fraud_Detection:

    # Logistic Regression Model
    def LogisticRegression_for_cc_fraud(self,filepath):
        """Logistic Regression model to predict 
        the Accuracy of a given Supervised Data
        Args: 
            dataset_name(str): Name of the csv file containing the data
        Return: 
            None
        """
        data = pd.read_csv(f"{filepath}.csv")
        
        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]
        
        # Under-Sampling: It's a technique to balance uneven datasets by keeping all of the data in the minority class and decreasing the size of the majority class
        if legit.shape > fraud.shape:
            legit_sample = legit.sample(n=len(fraud.index))
            # Concatinatng two df
            new_dataset = pd.concat([legit_sample, fraud],axis =0)
        elif legit.shape < fraud.shape:
            fraud_sample = fraud.sample(n=len(legit.index))
            # Concatinatng two df
            new_dataset = pd.concat([legit, fraud_sample],axis =0)
        else:
            new_dataset = data
        
        # Spliting the dataset into features and targets
        X = new_dataset.drop(columns='Class',axis=1)
        Y = new_dataset['Class']
        
        # Split the dataset into training data and test data
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=24)
        
        # Creating Logistic Regression Model object.
        model = LogisticRegression()
        
        # Fitting the model to the training data.
        model.fit(X_train,Y_train)
        
        # Accuracy score
        X_train_pred = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_pred,Y_train)
        # print(f"Accuracy score on training data : {training_data_accuracy*100}")
        return training_data_accuracy*100


    # Random Forest Model
    def random_forest_for_cc_fraud(self,filepath):
        """Random Forest Model for predicting fraud transactions in a given dataset

        Args:
            filepath (str): path with file name where the dataset is located

        Returns:
            float : Accuracy of the predictions
        """
        data = pd.read_csv(f"{filepath}.csv")

        # Spliting the dataset into features and targets
        X = data.drop('Class',axis=1)
        Y = data['Class']

        # Split the dataset into training data and test data
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

        # Create a random forest classifier object
        rfc = RandomForestClassifier(n_estimators=50)  # n_estimators : The number of trees in the forest

        # Train the model on the training data
        rfc.fit(X_train, Y_train)

        # Predict the test data using the trained model
        y_pred = rfc.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(Y_test, y_pred)

        return accuracy*100


    # Support Vector Machine(SVM) Model
    def SVM_for_cc_fraud(self,filepath):
        """Support Vector Machine(SVM) Model for predicting fraud transaction in a given dataset

        Args:
            filepath (str): path with file name where the dataset is located

        Returns:
            float : Accuracy of the predictions
        """
        data = pd.read_csv(f"{filepath}.csv")

        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]

        # Spliting the dataset into features and targets
        X = data.drop(columns='Class',axis=1)
        Y = data['Class']

        # Split the dataset into training data and test data
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

        SVM = SVC()
        SVM.fit(X_train,Y_train)

        # Predicting the values of the training data.
        X_pred = SVM.predict(X_test)
        # Calculating the accuracy of the model on the training data.
        training_data_accuracy = accuracy_score(X_pred,Y_test)
        
        return training_data_accuracy*100


# if __name__ == '__main__':

#     # # Creating an instance of the class CCD_Fraud_Detection.
#     CC = CCD_Fraud_Detection()
#     print(CC.LogisticRegression_for_cc_fraud())
#     print(CC.SVM_for_cc_fraud())
#     print(CC.random_forest_for_cc_fraud())
    # # Creating a gauge chart using plotly.
    # fig = go.Figure(go.Indicator(
    #     mode = "gauge+number",
    #     value = float(CC.LogisticRegression_for_cc_fraud()),
    #     domain = {'x': [0, 1], 'y': [0, 1]},
    #     title = {'text': "Accuracy Using Logistic Regression"},
    #     gauge = {'axis': { 'range': [None,100]}}))
    # fig.show()


    # # Creating a gauge chart using plotly.
    # fig = go.Figure(go.Indicator(
    #     mode = "gauge+number",
    #     value = CC.random_forest_for_cc_fraud(),
    #     domain = {'x': [0, 1], 'y': [0, 1]},
    #     title = {'text': "Accuracy Using Random Forest Model"},
    #     gauge = {'axis': { 'range': [None,100]}}))
    # fig.show()

    # Creating a gauge chart using plotly.
    # fig = go.Figure(go.Indicator(
    #     mode = "gauge+number",
    #     value = CC.SVM_for_cc_fraud(),
    #     domain = {'x': [0, 1], 'y': [0, 1]},
    #     title = {'text': "Accuracy Using SVM Model"},
    #     gauge = {'axis': { 'range': [None,100]}}))
    # fig.show()