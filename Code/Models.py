from Code import Preprocessing as cp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, cv
import pandas as pd
import numpy as np


class ModelAccidents():
    def __init__(self, df):
        self.crashes = df #get df from the file preprocessing.py

    def split_data(self, df):
        X = df.drop(['DAMAGE'], axis=1)
        y = df.DAMAGE

        print(X.dtypes)
        categorical_features_indices = np.where(X.dtypes != np.float)[0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y, shuffle=False)
        return X, y, X_train, X_test, y_train, y_test, categorical_features_indices

    def model_catboost(self, X, y, X_train, X_test, y_train, y_test, categorical_features_indices):
        model = CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy', use_best_model=True, random_seed=42) #iterations=200)
        model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_test, y_test), verbose=False, plot=True)
        print('CatBoost model is fitted: ' + str(model.is_fitted()))

        pred = model.predict(X_test)
        print(pred)

        cm = pd.DataFrame()
        cm['DAMAGE']= y_test
        cm['PREDICT'] = model.predict(X_test)
        print(cm)

        print('SCORES:')
        print(model.score(X_test, y_test))
        cm.to_csv("catboost_prediction.csv")


        return model

    def feature_importance(self):
        return False

