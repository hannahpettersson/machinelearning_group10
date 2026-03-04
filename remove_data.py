import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report



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


# remaining_nan_cat = new_df.isnull().sum().sum()

# #print(f"Remaining number of missing values categorical: {new_df[categorical_cols].isnull().sum().sum()}")
# #print(f"Total number of missing values: {c.BOLD}{remaining_nan_cat}{c.END}\n")
# #new_df.info()

# numerical_cols = new_df.select_dtypes(include=['number']).columns

# for num in numerical_cols:
#     if new_df[col].isnull().sum() > 0:
#         median = new_df[col].mean()
#         new_df[col] = new_df[col].fillna(median)

# remaining_nan_num = new_df.isnull().sum().sum()

# print(f"Remaining number of missing values numerical: {new_df[numerical_cols].isnull().sum().sum()}")
# print(f"Total number of missing values: {c.BOLD}{remaining_nan_num}{c.END}\n")
# new_df.info()


threshold = 0.06
min_valid_cols = int(len(new_df.columns) * (1 - threshold))
new_df = new_df.dropna(thresh=min_valid_cols)
print(f"Patients remaining: {len(new_df)}")


y = new_df['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
X_df = new_df.drop(columns=['readmitted'])

# Now encode only the features
encoded_df = pd.get_dummies(X_df, drop_first=True)

X = encoded_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a classifier
# Train a classifier
rand_forest_clf = RandomForestClassifier(
    n_estimators=200,        # more trees = more stable (default is only 100)
    max_depth=10,            # prevents overfitting on noisy diag_ columns
    min_samples_leaf=5,      # also helps with overfitting
    class_weight='balanced', # compensates for class imbalance automatically
    random_state=42          # makes results reproducible
)
rand_forest_clf.fit(X_train, y_train)
log_pred = rand_forest_clf.predict(X_test)
acc = rand_forest_clf.score(X_test, y_test)
print(f"Accuracy: {acc}")
print(classification_report(y_test, log_pred))


# Perform cross-validation 