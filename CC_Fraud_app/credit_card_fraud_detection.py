import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Using Different Models For CCD Fraud Detection
class CCD_Fraud_Detection:

    def __init__(self,filepath):
        self.filepath = filepath
        self.model_folder = "saved_models"


    def save_model(self, model, model_name):
        # Create the model folder if it does not exist
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        with open(f"{self.model_folder}/{model_name}.pkl", 'wb') as file:
            pickle.dump(model, file)


    def load_model(self, model_name):
        try:
            with open(f"{self.model_folder}/{model_name}.pkl", 'rb') as file:
                model = pickle.load(file)
                return model
        except FileNotFoundError:
            return None
        
    
    # Logistic Regression Model
    def LogisticRegression_for_cc_fraud(self):
        """Logistic Regression model to predict 
        the Accuracy of a given Supervised Data
        Args: 
            dataset_name(str): Name of the csv file containing the data
        Return: 
            None
        """
        if self.filepath == "creditcard":
            data = pd.read_csv(f"{self.filepath}.csv")
            X_original = data.drop(columns='Class',axis =1)
            Y_original = data['Class']
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
            X_train,_,Y_train,_ = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=24)
            _,X_test,_,Y_test =  train_test_split(X_original,Y_original,test_size=0.2,stratify=Y_original,random_state=24)
            
            # Scale the features before fitting the logistic regression model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = self.load_model("logistic_regression")
            if model is not None:
                # Accuracy score
                X_train_pred = model.predict(X_test_scaled)
                training_data_accuracy = accuracy_score(Y_test, X_train_pred)
                # print(f"Accuracy score on training data : {training_data_accuracy*100}")
                return training_data_accuracy*100, confusion_matrix(Y_test,X_train_pred)
            else:
                # Creating Logistic Regression Model object with an increased max_iter value
                model = LogisticRegression(max_iter=1000)

                # Fitting the model to the scaled training data.
                model.fit(X_train_scaled, Y_train)
                self.save_model(model, "logistic_regression")

                # Accuracy score
                X_train_pred = model.predict(X_test_scaled)
                training_data_accuracy = accuracy_score(Y_test,X_train_pred)
                # print(f"Accuracy score on training data : {training_data_accuracy*100}")
                return (training_data_accuracy*100), confusion_matrix(Y_test,X_train_pred)
        else:
            data = pd.read_csv(f"{self.filepath}.csv")
            X_original = data.drop(columns='is_fraud',axis =1)
            Y_original = data['is_fraud']
            legit = data[data.is_fraud == 0]
            fraud = data[data.is_fraud == 1]
            
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
            X = new_dataset.drop(columns='is_fraud',axis=1)
            Y = new_dataset['is_fraud']
            
            # Split the dataset into training data and test data
            X_train,_,Y_train,_ = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=24)
            _,X_test,_,Y_test =  train_test_split(X_original,Y_original,test_size=0.2,stratify=Y_original,random_state=24)
            
            # Scale the features before fitting the logistic regression model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = self.load_model("lr_processed_data")
            if model is not None:
                # Accuracy score
                X_train_pred = model.predict(X_test_scaled)
                training_data_accuracy = accuracy_score(Y_test, X_train_pred)
                # print(f"Accuracy score on training data : {training_data_accuracy*100}")
                return training_data_accuracy*100, confusion_matrix(Y_test,X_train_pred)
            else:
                # Creating Logistic Regression Model object with an increased max_iter value
                model = LogisticRegression(max_iter=1000)

                # Fitting the model to the scaled training data.
                model.fit(X_train_scaled, Y_train)
                self.save_model(model, "lr_processed_data")

                # Accuracy score
                X_train_pred = model.predict(X_test_scaled)
                training_data_accuracy = accuracy_score(Y_test,X_train_pred)
                # print(f"Accuracy score on training data : {training_data_accuracy*100}")
                return (training_data_accuracy*100), confusion_matrix(Y_test,X_train_pred)
        


    # Random Forest Model
    def random_forest_for_cc_fraud(self):
        """Random Forest Model for predicting fraud transactions in a given dataset

        Args:
            filepath (str): path with file name where the dataset is located

        Returns:
            float : Accuracy of the predictions
        """
        if self.filepath == 'creditcard':
            data = pd.read_csv(f"{self.filepath}.csv")

            # Spliting the dataset into features and targets
            X = data.drop('Class',axis=1)
            Y = data['Class']

            # Split the dataset into training data and test data
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

            model = self.load_model("random_forest")
            if model is not None:
                # Predict the test data using the trained model
                y_pred = model.predict(X_test)

                # Calculate the accuracy of the model
                accuracy = accuracy_score(Y_test, y_pred)
                return accuracy*100,confusion_matrix(Y_test, y_pred)

            else:
                # Create a random forest classifier object
                model = RandomForestClassifier(n_estimators=50)  # n_estimators : The number of trees in the forest

                # Train the model on the training data
                model.fit(X_train, Y_train)
                self.save_model(model,"random_forest")
                # Predict the test data using the trained model
                y_pred = model.predict(X_test)

                # Calculate the accuracy of the model
                accuracy = accuracy_score(Y_test, y_pred)

                return accuracy*100,confusion_matrix(Y_test, y_pred)
        else:
            data = pd.read_csv(f"{self.filepath}.csv")

            # Spliting the dataset into features and targets
            X = data.drop('is_fraud',axis=1)
            Y = data['is_fraud']

            # Split the dataset into training data and test data
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

            model = self.load_model("rf_processed_data")
            if model is not None:
                # Predict the test data using the trained model
                y_pred = model.predict(X_test)

                # Calculate the accuracy of the model
                accuracy = accuracy_score(Y_test, y_pred)
                return accuracy*100,confusion_matrix(Y_test, y_pred)

            else:
                # Create a random forest classifier object
                model = RandomForestClassifier(n_estimators=50)  # n_estimators : The number of trees in the forest

                # Train the model on the training data
                model.fit(X_train, Y_train)
                self.save_model(model,"rf_processed_data")
                # Predict the test data using the trained model
                y_pred = model.predict(X_test)

                # Calculate the accuracy of the model
                accuracy = accuracy_score(Y_test, y_pred)

                return accuracy*100,confusion_matrix(Y_test, y_pred)


    # Support Vector Machine(SVM) Model
    def SVM_for_cc_fraud(self):
        """Support Vector Machine(SVM) Model for predicting fraud transaction in a given dataset

        Args:
            filepath (str): path with file name where the dataset is located

        Returns:
            float : Accuracy of the predictions
        """
        if self.filepath == 'creditcard':
            data = pd.read_csv(f"{self.filepath}.csv")

            legit = data[data.Class == 0]
            fraud = data[data.Class == 1]

            # Spliting the dataset into features and targets
            X = data.drop(columns='Class',axis=1)
            Y = data['Class']

            # Split the dataset into training data and test data
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

            # Scale the features before fitting the SVM model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            SVM = self.load_model("support_vector_machines")
            if SVM is not None:
                X_pred = SVM.predict(X_test_scaled)
                training_data_accuracy = accuracy_score(Y_test, X_pred)
                return training_data_accuracy*100,confusion_matrix(Y_test, X_pred)
            else:    
                SVM = SVC()
                SVM.fit(X_train_scaled,Y_train)

                self.save_model(SVM,"support_vector_machines")
                
                # Predicting the values of the training data.
                X_pred = SVM.predict(X_test_scaled)
                # Calculating the accuracy of the model on the training data.
                training_data_accuracy = accuracy_score(Y_test,X_pred)
                
                return training_data_accuracy*100,confusion_matrix(Y_test, X_pred)
    
        else:
            data = pd.read_csv(f"{self.filepath}.csv")

            # Spliting the dataset into features and targets
            X = data.drop(columns='is_fraud',axis=1)
            Y = data['is_fraud']

            # Split the dataset into training data and test data
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

            # Scale the features before fitting the SVM model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            SVM = self.load_model("svm_processed_data")
            if SVM is not None:
                X_pred = SVM.predict(X_test_scaled)
                training_data_accuracy = accuracy_score(Y_test, X_pred)
                return training_data_accuracy*100,confusion_matrix(Y_test, X_pred)
            else:    
                SVM = SVC()
                SVM.fit(X_train_scaled,Y_train)

                self.save_model(SVM,"svm_processed_data")
                
                # Predicting the values of the training data.
                X_pred = SVM.predict(X_test_scaled)
                # Calculating the accuracy of the model on the training data.
                training_data_accuracy = accuracy_score(Y_test,X_pred)
                
                return training_data_accuracy*100,confusion_matrix(Y_test, X_pred)
