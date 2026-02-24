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
