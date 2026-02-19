import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Read/load the data: df = pd.read_csv('test_people.csv')

# Pre-process/format the data, if some values are null deal with them

# Convert values from categorical to numerical if applicable 

# Divide dataset and split into training and test set
# X = df.drop('class', axis=1)
# y = df['class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train a classifier
# log_reg_model = LogisticRegression(random_state=42)
# log_reg_model.fit(X_train, y_train)
# log_pred = log_reg_model.predict(X_test)
# acc = log_reg_model.score(X_test, y_test)
# print(f"Accuracy: {acc}")

# Perform cross-validation 


