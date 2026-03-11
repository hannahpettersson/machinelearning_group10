import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#applicera one hot?
#sannolikheten för någon som har högt eller numerisk ac1 result är väldigt hög att bli återinlagd

df = pd.read_csv('diabetic_data.csv')

new_df = df.copy()

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

# Preprocessing
new_df = df.drop(columns=['weight', 'encounter_id', 'patient_nbr', 'race'])

MEANINGFUL_THRESHOLD = 1.0  # Keep columns where >1% of values are meaningful

meaningful_pct = {}

for col in new_df.columns:
    total = len(new_df[col])
    is_not_placeholder = ~new_df[col].isin(['No', 'None', '?', 'Nan'])  # '?' is common in this dataset
    is_not_null = new_df[col].notna()
    
    meaningful_count = (is_not_placeholder & is_not_null).sum()
    meaningful_pct[col] = (meaningful_count / total) * 100

# Convert to Series so we can filter with a boolean mask
meaningful_pct = pd.Series(meaningful_pct)

cols_to_remove = meaningful_pct[meaningful_pct < MEANINGFUL_THRESHOLD].index.tolist()
print(f"Removing {len(cols_to_remove)} columns: {cols_to_remove}")

new_df = new_df.drop(columns=cols_to_remove)


remaining_nan_cat = new_df.isnull().sum().sum()

#print(f"Remaining number of missing values categorical: {new_df[categorical_cols].isnull().sum().sum()}")
#print(f"Total number of missing values: {c.BOLD}{remaining_nan_cat}{c.END}\n")
#new_df.info()

numerical_cols = new_df.select_dtypes(include=['number']).columns

for col in numerical_cols:
    if new_df[col].isnull().sum() > 0:
        median = new_df[col].median() 
        new_df[col] = new_df[col].fillna(median)

remaining_nan_num = new_df.isnull().sum().sum()

print(f"Remaining number of missing values numerical: {new_df[numerical_cols].isnull().sum().sum()}")
print(f"Total number of missing values: {c.BOLD}{remaining_nan_num}{c.END}\n")
new_df.info()


def parse_a1c(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    # Handle known text categories
    if val.lower() in ['none', '?']:
        return np.nan
    if val.lower() == 'norm':
        return 5.5  # Normal A1C approx
    # If purely numeric (e.g., 6, 7.2, 8)
    try:
        return float(val)
    except ValueError:
        pass
    # Handle >7, <6 etc.
    match = re.match(r'([<>])\s*(\d+\.?\d*)', val)
    if match:
        operator = match.group(1)
        number = float(match.group(2))
        if operator == '>':
            return number + 0.3
        elif operator == '<':
            return number - 0.3
    return np.nan 

new_df['a1c_numeric'] = new_df['A1Cresult'].apply(parse_a1c)
new_df = new_df.drop(columns=['A1Cresult'])

new_df['early_readmit'] = (new_df['readmitted'] == '<30').astype(int)

X_raw = new_df.drop(columns=['readmitted', 'early_readmit'])
y = new_df['early_readmit']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

X_train = pd.get_dummies(X_train_raw, drop_first=True)
X_test  = pd.get_dummies(X_test_raw,  drop_first=True)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

rand_forest_clf = RandomForestClassifier(class_weight='balanced')
rand_forest_clf.fit(X_train, y_train)

log_pred = rand_forest_clf.predict(X_test)
print(f"Accuracy: {rand_forest_clf.score(X_test, y_test)}")
print(classification_report(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))
