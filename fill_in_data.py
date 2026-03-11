import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('train.csv')

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

# print(f"The dataset length: \t\t{c.BLUE}{len(new_df)}{c.END}")
# print(f"Total number of missing values: {c.BOLD}{nan_total}{c.END}\n")
# print(f"Total length: {length}")
# print(f"{c.BOLD}Printing how many entries in each column contain no NaN values{c.END}:")
# new_df.info()

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

# print(f"Remaining number of missing values categorical: {new_df[categorical_cols].isnull().sum().sum()}")
# print(f"Total number of missing values: {c.BOLD}{remaining_nan_cat}{c.END}\n")
# new_df.info()

age_range = {'[0-10]': 5, '[10-20]': 15, '[20-30]': 25, '[30-40]': 35, '[40-50]': 45, 
             '[50-60]': 55, '[60-70]': 65, '[70-80]': 75, '[80-90]': 85, '[90-100]': 95}
new_df['age'] = new_df['age'].map(age_range)

numerical_cols = new_df.select_dtypes(include=['number']).columns
numerical_cols = numerical_cols.drop('age')

for num in numerical_cols:
    if new_df[num].isnull().sum() > 0:
        median = new_df[num].mean()
        new_df[num] = new_df[num].fillna(median)

remaining_nan_num = new_df.isnull().sum().sum()

# print(f"Remaining number of missing values numerical: {new_df[numerical_cols].isnull().sum().sum()}")
# print(f"Total number of missing values: {c.BOLD}{remaining_nan_num}{c.END}\n")
# new_df.info()

# Divide dataset and split into training and test set
X_unencoded = new_df.drop(columns=['readmitted', 'encounter_id','patient_nbr'], axis=1)
y = new_df['readmitted']

# Convert values from categorical to numerical if applicable 
X = pd.get_dummies(X_unencoded, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a classifier
rand_forest_clf = RandomForestClassifier()
rand_forest_clf.fit(X_train, y_train)
log_pred = rand_forest_clf.predict(X_test)
acc = rand_forest_clf.score(X_test, y_test)
print(f"Accuracy train: {acc}")

#----------------------------------------------------------#

d_test = pd.read_csv('test.csv')

new_df_test = d_test.copy()

nan_values_per_feature = new_df_test.isnull().sum()
nan_total = sum(list(new_df_test.isnull().sum()))
length = len(df)

# Preprocessing
new_df_test = d_test.drop(columns=['weight'], columns=['payer_code'])
new_df_test.replace('?', np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')

categorical_cols = new_df_test.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if new_df_test[col].isnull().sum() > 0:
        most_frequent = new_df_test[col].mode()[0]
        new_df_test[col] = new_df_test[col].fillna(most_frequent)

remaining_nan_cat = new_df_test.isnull().sum().sum()

age_range = {'[0-10]': 5, '[10-20]': 15, '[20-30]': 25, '[30-40]': 35, '[40-50]': 45, 
             '[50-60]': 55, '[60-70]': 65, '[70-80]': 75, '[80-90]': 85, '[90-100]': 95}
new_df_test['age'] = new_df_test['age'].map(age_range)

numerical_cols = new_df_test.select_dtypes(include=['number']).columns
numerical_cols = numerical_cols.drop('age')

for num in numerical_cols:
    if new_df_test[num].isnull().sum() > 0:
        median = new_df_test[num].mean()
        new_df_test[num] = new_df_test[num].fillna(median)

remaining_nan_num = new_df_test.isnull().sum().sum()

# Divide dataset and split into training and test set
X_unencoded_test_data = new_df_test.drop(columns=['encounter_id','patient_nbr'], axis=1)
# y_test_data = new_df_test['readmitted']

# Convert values from categorical to numerical if applicable 
X_test_data = pd.get_dummies(X_unencoded_test_data, drop_first=True)

X_final_test = X_test_data.reindex(columns=X_train.columns, fill_value=0)

# 4. Predict!
final_preds = rand_forest_clf.predict(X_final_test)
# acc_test = rand_forest_clf.score(X_final_test, y_test)
# print(f"Final test accuracy: {acc_test}")

# 5. Create the submission file
# Based on the diabetic dataset, the ID column is usually 'encounter_id'
submission = pd.DataFrame({
    'id': new_df_test['id'], 
    'readmitted': final_preds
})

submission.to_csv('submission.csv', index=False)
print(f"{c.GREEN}✅ submission.csv created successfully!{c.END}")















# # 1. Load the actual test set
# final_test_df = pd.read_csv('test.csv')

# # 2. Preprocess final_test_df EXACTLY like the train set
# # We create a copy to avoid modifying the original dataframe needed for IDs
# test_proc = final_test_df.copy()

# # A. Drop weight and replace '?'
# test_proc = test_proc.drop(columns=['weight'])
# test_proc.replace('?', np.nan, inplace=True)

# # B. Map Age (Same as training)
# test_proc['age'] = test_proc['age'].map(age_range)

# # C. Fill NaNs for Categorical
# cat_cols_test = test_proc.select_dtypes(include=['object']).columns
# for col in cat_cols_test:
#     # Use the mode from training if possible, but for simplicity here we use test mode
#     if test_proc[col].isnull().sum() > 0:
#         test_proc[col] = test_proc[col].fillna(test_proc[col].mode()[0])

# # D. Fill NaNs for Numerical
# num_cols_test = test_proc.select_dtypes(include=['number']).columns
# for num in num_cols_test:
#     if test_proc[num].isnull().sum() > 0:
#         test_proc[num] = test_proc[num].fillna(test_proc[num].mean())

# # E. Drop ID columns and Encode
# X_final_test_unencoded = test_proc.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')
# X_final_test = pd.get_dummies(X_final_test_unencoded, drop_first=True)

# # 3. Align columns (Crucial! Ensures test features match training features)
# X_final_test = X_final_test.reindex(columns=X_train.columns, fill_value=0)

# # 4. Predict!
# final_preds = rand_forest_clf.predict(X_final_test)

# # 5. Create the submission file
# # Based on the diabetic dataset, the ID column is usually 'encounter_id'
# submission = pd.DataFrame({
#     'id': final_test_df['id'], 
#     'readmitted': final_preds
# })

# submission.to_csv('submission.csv', index=False)
# print(f"{c.GREEN}✅ submission.csv created successfully!{c.END}")

# Perform cross-validation 
# svc_clf = LinearSVC(random_state=42)
# dec_tree_clf = DecisionTreeClassifier(random_state=42)
# rand_forest_clf = RandomForestClassifier(random_state=42)

# clsf = [svc_clf, dec_tree_clf, rand_forest_clf]

# names = ["SVC", "Decision Tree", "Random Forest"]

# for i in range(len(clsf)):
#     cross_val_res = cross_val_score(estimator=clsf[i], X=X, y=y, cv=10)
#     print(f"{names[i]} Average accuracy: {cross_val_res.mean():.4f}")

