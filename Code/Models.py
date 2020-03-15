from Code import Preprocessing as cp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, cv
import pandas as pd
import numpy as np


class ModelAccidents():
    def __init__(self):
        self.crashes = cp.df_crashes #get df from the file preprocessing.py

    def encode_cols(self, df):
        return pd.get_dummies(df, columns=['TRAFFIC_CONTROL_DEVICE','DEVICE_CONDITION', 'WEATHER_CONDITION',
                                           'LIGHTING_CONDITION', 'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE',
                                           'ALIGNMENT', 'ROADWAY_SURFACE_COND', 'ROAD_DEFECT', 'CRASH_TYPE', 'DAMAGE',
                                           'PRIM_CONTRIBUTORY_CAUSE', 'SEC_CONTRIBUTORY_CAUSE', 'STREET_DIRECTION',
                                           'STREET_NAME', 'MOST_SEVERE_INJURY', 'PERSON_TYPE', 'STATE', 'SEX',
                                           'AIRBAG_DEPLOYED', 'EJECTION', 'INJURY_CLASSIFICATION',	'DRIVER_ACTION',
                                           'DRIVER_VISION', 'PHYSICAL_CONDITION', 'UNIT_TYPE',	'MAKE',	'VEHICLE_YEAR',
                                           'VEHICLE_DEFECT', 'VEHICLE_TYPE', 'OCCUPANT_CNT'])

    def split_data(self, df):
        X = df.drop(['DAMAGE'], axis=1)
        y = df.DAMAGE
        print(X.dtypes)
        categorical_features_indices = np.where(X.dtypes != np.float)[0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y, shuffle=False)
        return X_train, X_test, y_train, y_test, categorical_features_indices

    def model_catboost(self, X_train, X_test, y_train, y_test, categorical_features_indices):
        # modify iterations
        model = CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy', use_best_model=True,
                                   random_seed=42, iterations=200)
        model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_test, y_test), verbose=False, plot=True)
        print('CatBoost model is fitted: ' + str(model.is_fitted()))
        return model

    def feature_importance(self):
        return False

