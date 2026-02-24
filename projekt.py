import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('diabetic_data.csv', index_col=0, na_values="?")
newdf = df.copy()

nan_values_per_feature = newdf.isnull().sum()
nan_total = sum(list(newdf.isnull().sum()))
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
    
print(f"The dataset length: \t\t{c.BLUE}{len(newdf)}{c.END}")
print(f"Total number of missing values: {c.BOLD}{nan_total}{c.END}\n")
print(f"Total length: {length}")
print(f"{c.BOLD}Printing how many entries in each column contain no NaN values{c.END}:")
newdf.info()

ages = newdf["age"]
plt.hist(ages, bins=20)
plt.xlabel("Age")
plt.ylabel("Quantity")
plt.legend()
plt.show()

readmitted = df.columns[48]
read_labels = {'NO': 0, '>30': 1, '<30': 2}
df[readmitted] = df[readmitted].map(read_labels)
plt.hist(df[readmitted], bins = 20)
plt.title("Plot over the readmitted")
plt.xlabel("Readmitted")
plt.ylabel("Quantity")



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

