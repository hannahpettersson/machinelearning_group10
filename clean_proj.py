import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
from imblearn.under_sampling import RandomUnderSampler
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


#-------------------------------TRAIN-----------------------------------#
df = pd.read_csv('train.csv')
new_df = df.copy()

# PREPROCESSING (FEATURE CREATION AND DROP) #
new_df = new_df.drop_duplicates(subset='patient_nbr', keep='first')
new_df = new_df.drop(columns=['weight', 'payer_code', 'medical_specialty', 'encounter_id','patient_nbr'])
new_df.replace('?', np.nan, inplace=True)

expired_ids = [11, 13, 14, 19, 20, 21]
new_df = new_df[~new_df['discharge_disposition_id'].isin(expired_ids)]

## Feature Creation ##
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

diag_cols = ['diag_1', 'diag_2', 'diag_3']

new_df['procs_per_day'] = new_df['num_lab_procedures'] / (new_df['time_in_hospital'] + 1)
new_df['meds_per_day'] = new_df['num_medications'] / (new_df['time_in_hospital'] + 1)
new_df['total_visits'] = new_df['number_outpatient'] + new_df['number_emergency'] + new_df['number_inpatient']
new_df['visits_per_hospital_day'] = new_df['total_visits'] / (new_df['time_in_hospital'] + 1)
new_df['num_comorbidities'] = new_df[diag_cols].apply(lambda x: sum(x != 'Other'), axis=1)
new_df['polypharmacy'] = (new_df['num_medications'] > 10).astype(int)
new_df['has_diabetes'] = new_df[diag_cols].apply(lambda x: int('Diabetes' in x.values), axis=1)
new_df['has_cardiovascular'] = new_df[diag_cols].apply(lambda x: int('Circulatory' in x.values), axis=1)
new_df['A1Cresult'] = new_df['A1Cresult'].apply(parse_a1c)

## Handling categorical and NaN values ##
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

age_range = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, 
             '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
new_df['age'] = new_df['age'].map(age_range)

categorical_cols = new_df.select_dtypes(include=['object']).columns

categorical_cols = [
    col for col in new_df.select_dtypes(include=['object']).columns
    if col not in diag_cols
]

categorical_frequent = {}
for col in categorical_cols:
    if new_df[col].isnull().sum() > 0:
        most_frequent = new_df[col].mode()[0]
        categorical_frequent[col] = most_frequent
        new_df[col] = new_df[col].fillna(most_frequent)

diag_frequent = {}

for col in diag_cols:
    new_df[col] = new_df[col].fillna('Missing').apply(map_icd9)
    freq = new_df[col].value_counts(normalize=True)
    diag_frequent[col] = freq
   # 
   # new_df[col] = new_df[col].map(freq)

new_df['diag_severity'] = new_df[diag_cols].apply(lambda row: sum(row.map(lambda x: diag_frequent.get(x, 0))), axis=1)


## Handling numerical and NaN values ##
numerical_cols = new_df.select_dtypes(include=['number']).columns
numerical_cols = numerical_cols.drop('age')

numerical_frequent = {}
for num in numerical_cols:
    if new_df[num].isnull().sum() > 0:
        median = new_df[num].mean()
        numerical_frequent[num] = median
        new_df[num] = new_df[num].fillna(median)

# TRAINING THE MODEL # 
X_unencoded = new_df.drop(columns=['readmitted'])
y = new_df['readmitted']
y = y.apply(lambda x: 'Yes' if x == '<30' or x == '>30' else 'No')

X = pd.get_dummies(X_unencoded, drop_first=True)

## Splitting the data ##
custom_weights = {'Yes': 2, 'No': 1}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rand_forest_clf = RandomForestClassifier(n_estimators=400, max_depth=10)


rus = RandomUnderSampler(sampling_strategy='not minority', random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
rand_forest_clf.fit(X_train_res, y_train_res)
train_pred = rand_forest_clf.predict(X_test)

#-------------------------------TEST-----------------------------------#
d_test = pd.read_csv('test.csv')
new_df_test = d_test.copy()


# PREPROCESSING AND FEATURE CREATION: TEST DATA #
#new_df_test = new_df_test.drop_duplicates(subset='patient_nbr', keep='first')
new_df_test = new_df_test.drop(columns=['weight', 'payer_code', 'medical_specialty', 'encounter_id','patient_nbr'])

# expired_ids = [11, 13, 14, 19, 20, 21] alredy defined in train
#new_df_test = new_df_test[~new_df_test['discharge_disposition_id'].isin(expired_ids)]

## Feature Creation ##
new_df_test['procs_per_day'] = new_df_test['num_lab_procedures'] / (new_df_test['time_in_hospital'] + 1)
new_df_test['meds_per_day'] = new_df_test['num_medications'] / (new_df_test['time_in_hospital'] + 1)
new_df_test['total_visits'] = new_df_test['number_outpatient'] + new_df_test['number_emergency'] + new_df_test['number_inpatient']
new_df_test['visits_per_hospital_day'] = new_df_test['total_visits'] / (new_df_test['time_in_hospital'] + 1)
new_df_test['num_comorbidities'] = new_df_test[diag_cols].apply(lambda x: sum(x != 'Other'), axis=1)
new_df_test['diag_severity'] = new_df_test[diag_cols].apply(lambda row: sum(row.map(lambda x: diag_frequent.get(x, 0))), axis=1)
new_df_test['polypharmacy'] = (new_df_test['num_medications'] > 10).astype(int)
new_df_test['has_diabetes'] = new_df_test[diag_cols].apply(lambda x: int('Diabetes' in x.values), axis=1)
new_df_test['has_cardiovascular'] = new_df_test[diag_cols].apply(lambda x: int('Circulatory' in x.values), axis=1)
new_df_test['A1Cresult'] = new_df_test['A1Cresult'].apply(parse_a1c)

## Handling categorical and NaN data: TEST DATA ##
new_df_test['age'] = new_df_test['age'].map(age_range)

for col in categorical_cols:
    if col == 'readmitted':
        continue

    if col in categorical_frequent:
        new_df_test[col] = new_df_test[col].fillna(categorical_frequent[col])
    else:
        new_df_test[col] = new_df_test[col].fillna(new_df[col].mode()[0])


for col in diag_cols:
    new_df_test[col] = new_df_test[col].fillna('Missing')
    new_df_test[col] = new_df_test[col].apply(map_icd9)


## Handling numerical and NaN data: TEST DATA ##
for num in numerical_cols:
    if num == 'readmitted':
        continue
        
    if num in numerical_frequent:
        new_df_test[num] = new_df_test[num].fillna(numerical_frequent[num])
    else:
        new_df_test[num] = new_df_test[num].fillna(new_df[num].mean())

# PREDICTING: TEST DATA #
X_test_data = pd.get_dummies(new_df_test, drop_first=True)
X_final_test = X_test_data.reindex(columns=X_train.columns, fill_value=0)
final_preds = rand_forest_clf.predict(X_final_test)

submission = pd.DataFrame({
    'id': new_df_test['id'], 
    'readmitted': final_preds
})

submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created successfully!")

print(y.value_counts(normalize=True)) 
print(classification_report(y_test, train_pred))

# CONFUSION MATRIX #


cm = confusion_matrix(y_test, train_pred, labels=['Yes', 'No'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Yes', 'No'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()


# Feature importance
importances = rand_forest_clf.feature_importances_
feat_names = X_train.columns
feat_imp = pd.Series(importances, index=feat_names)
top_feats = feat_imp.sort_values(ascending=False).head(15)
top_feats.plot(kind='barh')
plt.title("Top 15 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Classification report
report = classification_report(y_test, train_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report[['precision','recall','f1-score']].iloc[:-1].plot(kind='bar')
plt.title("Metrics per Class")
plt.xticks(rotation=0)
plt.show()

# distribution of classes in the original data
order = ['No', 'Yes']
percentages = y.value_counts(normalize=True).reindex(order) * 100
sns.barplot(x=percentages.index, y=percentages.values)
plt.title("Distribution of Readmission Classes (%)")
plt.xlabel("Readmission Category")
plt.ylabel("Percentage (%)")
for i, v in enumerate(percentages.values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.show()
