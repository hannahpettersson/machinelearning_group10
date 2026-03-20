import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import re

#-------------------------------TRAIN-----------------------------------#

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

# Preprocessing

def map_icd9(code):
    if pd.isnull(code) or code == '?' or code == 'None':
        return 'Other'
    
    # Ibland innehåller koder bokstäver (V eller E), vi sätter dem som 'Other'
    try:
        # Om koden innehåller punkt (t.ex. 250.01), ta bara siffran före punkten
        val = float(str(code).split('.')[0])
    except ValueError:
        return 'Other'

    if 390 <= val <= 459 or val == 785:
        return 'Circulatory'
    elif 460 <= val <= 519 or val == 786:
        return 'Respiratory'
    elif 520 <= val <= 579 or val == 787:
        return 'Digestive'
    elif val == 250:
        return 'Diabetes'
    elif 800 <= val <= 999:
        return 'Injury'
    elif 710 <= val <= 739:
        return 'Musculoskeletal'
    elif 580 <= val <= 629 or val == 788:
        return 'Genitourinary'
    elif 140 <= val <= 239:
        return 'Neoplasms'
    else:
        return 'Other'
    
#AC1
def parse_a1c(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val.lower() in ['none', '?']:
        return np.nan
    if val.lower() == 'norm':
        return 5.5
    try:
        return float(val)
    except ValueError:
        pass
    match = re.match(r'([<>])\s*(\d+\.?\d*)', val)
    if match:
        operator = match.group(1)
        number = float(match.group(2))
        if operator == '>':
            return number + 0.3
        elif operator == '<':
            return number - 0.3
    return np.nan


new_df = df.drop(columns=['weight', 'payer_code', 'medical_specialty'])
new_df = new_df.drop_duplicates(subset='patient_nbr', keep='first')
#
new_df['procs_per_day'] = new_df['num_lab_procedures'] / (new_df['time_in_hospital'] + 1)
new_df['meds_per_day'] = new_df['num_medications'] / (new_df['time_in_hospital'] + 1)
new_df['total_visits'] = new_df['number_outpatient'] + new_df['number_emergency'] + new_df['number_inpatient']
new_df.replace('?', np.nan, inplace=True)
new_df['A1Cresult'] = new_df['A1Cresult'].apply(parse_a1c) #A1C

#patienter som dött eller flyttats till hospice
expired_ids = [11, 13, 14, 19, 20, 21]
new_df = new_df[~new_df['discharge_disposition_id'].isin(expired_ids)]


change_to_string = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
for i in range(len(change_to_string)):
    new_df[change_to_string[i]] = new_df[change_to_string[i]].astype(str)

age_range = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, 
             '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
new_df['age'] = new_df['age'].map(age_range)

numerical_cols = new_df.select_dtypes(include=['number']).columns
numerical_cols = numerical_cols.drop('age')

numerical_frequent = {}
for num in numerical_cols:
    if new_df[num].isnull().sum() > 0:
        median = new_df[num].mean()
        numerical_frequent[num] = median
        new_df[num] = new_df[num].fillna(median)


diag_cols = ['diag_1', 'diag_2', 'diag_3']
for col in diag_cols:
    new_df[col] = new_df[col].apply(map_icd9)

categorical_cols = new_df.select_dtypes(include=['object']).columns
categorical_frequent = {}
for col in categorical_cols:
    if new_df[col].isnull().sum() > 0:
        most_frequent = new_df[col].mode()[0]
        categorical_frequent[col] = most_frequent
        new_df[col] = new_df[col].fillna(most_frequent)

# Divide dataset and split into training and test set
X_unencoded = new_df.drop(columns=['readmitted', 'encounter_id','patient_nbr', 'id'], axis=1)
y = new_df['readmitted']


# Convert values from categorical to numerical if applicable 
X = pd.get_dummies(X_unencoded, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a classifier
rand_forest_clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=20)
rand_forest_clf.fit(X_train, y_train)

importances = rand_forest_clf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
print(feat_importances.nlargest(10)) # Visar de 10 viktigaste kolumnerna

log_pred = rand_forest_clf.predict(X_test)
acc = rand_forest_clf.score(X_test, y_test)
print(f"Accuracy train: {acc}")

#-------------------------------TEST-----------------------------------#
d_test = pd.read_csv('test.csv')

new_df_test = d_test.copy()

# nan_values_per_feature = new_df_test.isnull().sum()
# nan_total = sum(list(new_df_test.isnull().sum()))
# length = len(df)

# Preprocessing
new_df_test = d_test.drop(columns=['weight', 'payer_code', 'medical_specialty'])

new_df_test['procs_per_day'] = new_df_test['num_lab_procedures'] / (new_df_test['time_in_hospital'] + 1)
new_df_test['meds_per_day'] = new_df_test['num_medications'] / (new_df_test['time_in_hospital'] + 1)

new_df_test['total_visits'] = new_df_test['number_outpatient'] + new_df_test['number_emergency'] + new_df_test['number_inpatient']
new_df_test.replace('?', np.nan, inplace=True)
new_df_test['A1Cresult'] = new_df_test['A1Cresult'].apply(parse_a1c) #A1c

#new_df_test = new_df.drop_duplicates(subset='patient_nbr', keep='first')

# NYTT
for col in diag_cols:
    new_df_test[col] = new_df_test[col].apply(map_icd9)

new_df_test['age'] = new_df_test['age'].map(age_range)

for i in range(len(change_to_string)):
    new_df_test[change_to_string[i]] = new_df_test[change_to_string[i]].astype(str)

for num in numerical_cols:
    if num == 'readmitted':
        continue
        
    if num in numerical_frequent:
        new_df_test[num] = new_df_test[num].fillna(numerical_frequent[num])
    else:
        # Fallback: if the column had no NaNs in train, use the mean of the column now
        new_df_test[num] = new_df_test[num].fillna(new_df[num].mean())

for col in categorical_cols:
    if col == 'readmitted':
        continue

    if col in categorical_frequent:
        new_df_test[col] = new_df_test[col].fillna(categorical_frequent[col])
    else:
        # Fallback for columns that were clean in training but messy in test
        new_df_test[col] = new_df_test[col].fillna(new_df[col].mode()[0])

# 5. Prepare features for prediction
# Drop IDs (but keep encounter_id or id elsewhere for the submission!)

#kaggle
# Divide dataset and split into training and test set
X_unencoded_test_data = new_df_test.drop(columns=['encounter_id','patient_nbr', 'id'], axis=1)
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

print(y.value_counts(normalize=True))  # Check class balance
print(feat_importances.nlargest(15))   # See where A1C ranks
print(classification_report(y_test, log_pred))