#
#DROPOUT - 0 GRADUATE -1 
# libraries and requirements
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from yellowbrick.classifier import confusion_matrix
import imblearn
from imblearn.over_sampling import SMOTE 
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score

# we are only going to make predictions with balanced data
dataset_raw = pd.read_csv('/Users/angadbawa/Development/Dep/UCIdataset/realData - data.csv')


# create copy of dataset
dataset = dataset_raw.copy()
dataset.shape

# getting rid of enrolled data
dataset.drop(dataset[dataset["Target"] == "Enrolled"].index, inplace=True)

# eliminating unnecessary columns
cols_eliminate = ['Marital status', 'Application order', 'Course', 'Curricular units 1st sem (grade)',
                  'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
                  'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
                  'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (grade)',
                  'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
                  'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
                  'Curricular units 2nd sem (without evaluations)', 'Previous qualification (grade)', 'Admission grade']
dataset.drop(columns=[col for col in cols_eliminate if col in dataset.columns], inplace=True)

target = dataset['Target']
features = dataset.drop(['Target'], axis=1)
target.shape, features.shape

#label encoding ( 0 droput 1 graduate )

le = LabelEncoder()
target = le.fit_transform(target)

# Use SMOTE
oversample = SMOTE()
features_2, target_2 = oversample.fit_resample(features, target)

# Summarize distribution
counter = Counter(target_2)

# Plot the distribution
plt.figure(figsize=(3, 3))
plt.bar(counter.keys(), counter.values())
plt.show()

# Trained on balanced data
X_train, X_test, y_train, y_test = train_test_split(features_2, target_2, test_size=0.2, random_state=2304)



model = XGBClassifier(n_estimators=100, random_state=2304, eval_metric='mlogloss', use_label_encoder=False)
model.fit(features_2, target_2)

# Single tuple to classify
X_tuple = pd.DataFrame({
    'Application mode': [1],
    'Daytime/evening attendance': [1],
    'Previous qualification': [1],
    'Nacionality': [1],
    "Mother's qualification": [37],
    "Father's qualification": [37],
    "Mother's occupation": [9],
    "Father's occupation": [9],
    'Displaced': [1],
    'Educational special needs': [0],
    'Debtor': [0],
    'Tuition fees up to date': [0],
    'Gender': [1],
    'Scholarship holder': [0],
    'Age at enrollment': [19],
    'International': [0],
    'Unemployment rate': [10.8],
    'Inflation rate': [1.4],
    'GDP': [1.74]
})

# Predict the class label of the single tuple
pred_label = model.predict(X_tuple)

print("Predicted label:", pred_label)

# Accuracy score on X_test
pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, pred_test)
print("Accuracy score on X_test:", accuracy)


import pickle 
pickle.dump(model,open('xgbmodel.pkl','wb'))

model = pickle.load(open('xgbmodel.pkl','rb'))
print(model.predict(X_tuple))

