import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset into a Pandas DataFrame
credit_card_data = pd.read_csv('/content/credit_data.csv')

# Display the first 5 rows of the dataset
print("First 5 rows:")
print(credit_card_data.head())

# Display the last 5 rows of the dataset
print("Last 5 rows:")
print(credit_card_data.tail())

# Display dataset information (data types, non-null counts, etc.)
print("Dataset info:")
print(credit_card_data.info())

# Check the number of missing values in each column
print("Missing values:")
print(credit_card_data.isnull().sum())

# Display the distribution of legitimate and fraudulent transactions
print("Distribution of transactions:")
print(credit_card_data['Class'].value_counts())

# Separate the data into legitimate and fraudulent transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print("Legitimate transactions shape:", legit.shape)
print("Fraudulent transactions shape:", fraud.shape)

# Display statistical measures of the transaction amounts
print("Statistical measures for legitimate transactions:")
print(legit.Amount.describe())
print("Statistical measures for fraudulent transactions:")
print(fraud.Amount.describe())

# Compare the average values for both transaction types
print("Average values by Class:")
print(credit_card_data.groupby('Class').mean())

# Create a balanced dataset by sampling from the legitimate transactions
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Display the first 5 rows of the new dataset
print("First 5 rows of the new dataset:")
print(new_dataset.head())

# Display the last 5 rows of the new dataset
print("Last 5 rows of the new dataset:")
print(new_dataset.tail())

# Display the distribution of the classes in the new dataset
print("Class distribution in the new dataset:")
print(new_dataset['Class'].value_counts())

# Display the average values for both transaction types in the new dataset
print("Average values by Class in the new dataset:")
print(new_dataset.groupby('Class').mean())

# Prepare features and target variable
X = new_dataset.drop(columns='Class', axis=1)  # Features
Y = new_dataset['Class']  # Target variable
print("Target variable:")
print(Y)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("Shapes of the datasets:")
print("X shape:", X.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the Logistic Regression model with training data
model.fit(X_train, Y_train)

# Predict and evaluate the model on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on Training data:', training_data_accuracy)

# Predict and evaluate the model on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on Test Data:', test_data_accuracy)
