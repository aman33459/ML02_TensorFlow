import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
from collections import defaultdict
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


data1 = pd.read_csv('/home/aman/Downloads/prog/hack_test/ensemble/train.csv')
data2 = pd.read_csv('/home/aman/Downloads/prog/hack_test/ensemble/test.csv')
print(data1)
y_train = data1['Survived']
del data1['Survived']
X_train_full, X_valid_full, y_train, y_valid = train_test_split(data1, y_train,
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])





numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = data2[my_cols].copy()


X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train , y_train , early_stopping_rounds=20, eval_set=[(X_valid,y_valid)])
preds = my_model.predict(X_test)
output = pd.DataFrame({'PassengerId': X_test.PassengerId,
                       'Survived': preds})
output.to_csv('/home/aman/Downloads/prog/hack_test/ensemble1.csv', index=False)


from sklearn.metrics import mean_absolute_error
model = RandomForestClassifier(n_estimators=100,criterion='entropy')
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])


clf1.fit(X_train, y_train ,model__early_stopping_rounds=20, model__eval_set=[(X_valid,y_valid)])


clf.fit(X_train, y_train)
preds = clf.predict(X_valid)
print(preds)
print('MAE:', mean_absolute_error(y_valid, preds))
predicted_val = clf.predict(X_test)
print(predicted_val)
output = pd.DataFrame({'PassengerId': X_test.PassengerId,
                       'Survived': predicted_val})
output.to_csv('/home/aman/Downloads/prog/hack_test/ensemble1.csv', index=False)
