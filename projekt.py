import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

df = pd.read_csv('diabetic_data.csv')

new_df = df.copy()

nan_values_per_feature = new_df.isnull().sum()
nan_total = sum(list(new_df.isnull().sum()))
length = len(df)

class c:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(f"The dataset length: \t\t{c.BLUE}{len(new_df)}{c.END}")
print(f"Total number of missing values: {c.BOLD}{nan_total}{c.END}\n")
print(f"Total length: {length}")
print(f"{c.BOLD}Printing how many entries in each column contain no NaN values{c.END}:")
new_df.info()

# Preprocessing
new_df = df.drop(columns=['weight'])
new_df.replace('?', np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')

categorical_cols = new_df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if new_df[col].isnull().sum() > 0:
        most_frequent = new_df[col].mode()[0]
        new_df[col] = new_df[col].fillna(most_frequent)

remaining_nan_cat = new_df.isnull().sum().sum()

print(f"Remaining number of missing values categorical: {new_df[categorical_cols].isnull().sum().sum()}")
print(f"Total number of missing values: {c.BOLD}{remaining_nan_cat}{c.END}\n")
new_df.info()

numerical_cols = new_df.select_dtypes(include=['number']).columns

for num in numerical_cols:
    if new_df[col].isnull().sum() > 0:
        median = new_df[col].mean()
        new_df[col] = new_df[col].fillna(median)

remaining_nan_num = new_df.isnull().sum().sum()

print(f"Remaining number of missing values numerical: {new_df[numerical_cols].isnull().sum().sum()}")
print(f"Total number of missing values: {c.BOLD}{remaining_nan_num}{c.END}\n")
new_df.info()

# Convert values from categorical to numerical if applicable 
encoded_df = pd.get_dummies(new_df, drop_first=True)

# Divide dataset and split into training and test set
X = encoded_df.drop('readmitted', axis=1)
y = encoded_df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a classifier
rand_forest_clf = RandomForestClassifier()
rand_forest_clf.fit(X_train, y_train)
log_pred = rand_forest_clf.predict(X_test)
acc = rand_forest_clf.score(X_test, y_test)
print(f"Accuracy: {acc}")

# Perform cross-validation 


