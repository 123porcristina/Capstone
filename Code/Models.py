from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool, cv
import pandas as pd
import numpy as np


class ModelAccidents():
    def __init__(self, df):
        self.crashes = df #get df from the file preprocessing.py

    def split_data(self):
        X = self.df.drop(['DAMAGE'], axis=1)
        y = self.df.DAMAGE

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

    def dict_classifiers(self):
        clfs = {'lr': LogisticRegression(random_state=0),
                'mlp': MLPClassifier(random_state=0),
                'cb': CatBoostClassifier(random_state=0),
                'rf': RandomForestClassifier(random_state=0),
                'xgb': XGBClassifier(seed=0),
                'svc': SVC(random_state=0),
                'knn': KNeighborsClassifier()}
        param_grids = {}
        return clfs, param_grids

    def dict_pipeline(self, clfs):
        pipe_clfs = {}

        for name, clf in clfs.items():
            pipe_clfs[name] = Pipeline(
                [('StandardScaler', StandardScaler()), ('clf', clf)])  # ([('prep', col_transform), ('clf', clf)])

        print(pipe_clfs['mlp'])
        return pipe_clfs

    def params_lr(self, param_grids):

        C_range = [10 ** i for i in range(-4, 5)]

        param_grid = [{'clf__multi_class': ['ovr'],
                       'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                       'clf__C': C_range},

                      {'clf__multi_class': ['multinomial'],
                       'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                       'clf__C': C_range}]

        param_grids['lr'] = param_grid
        return param_grids

    def params_mlp(self, param_grids):
        param_grid = [{'clf__hidden_layer_sizes': [10, 100],
                       'clf__activation': ['identity', 'logistic', 'tanh', 'relu']}]

        param_grids['mlp'] = param_grid
        return param_grids

    def params_rf(self, param_grids):
        param_grid = [{'clf__n_estimators': [10, 100],
                       'clf__min_samples_split': [2, 10, 30],
                       'clf__min_samples_leaf': [1, 10, 30]}]

        param_grids['rf'] = param_grid
        return param_grids

    def params_xgboost(self, param_grids):
        param_grid = [{'clf__eta': [10 ** i for i in range(-4, 1)],
                       'clf__gamma': [0, 10, 100],
                       'clf__lambda': [10 ** i for i in range(-4, 5)]}]

        param_grids['xgb'] = param_grid
        return param_grids

    def params_svc(self, param_grids):
        param_grid = [{'clf__C': [10 ** i for i in range(-4, 5)],
                       'clf__gamma': ['auto', 'scale']}]

        param_grids['svc'] = param_grid
        return param_grids

    def params_knn(self, param_grids):
        param_grid = [{'clf__n_neighbors': list(range(1, 11))}]

        param_grids['knn'] = param_grid
        return param_grids

    def hyperparameter_tunning(self, pipe_clfs, param_grids, X_train, y_train):
        # The list of [best_score_, best_params_, best_estimator_]
        best_score_param_estimators = []

        # For each classifier
        for name in pipe_clfs.keys():
            # GridSearchCV
            # Implement me
            gs = GridSearchCV(estimator=pipe_clfs[name],
                              param_grid=param_grids[name],
                              scoring='accuracy',
                              n_jobs=1,
                              iid=False,
                              cv=StratifiedKFold(n_splits=5,
                                                 shuffle=True,
                                                 random_state=0))
            # Fit the pipeline
            # Implement me
            gs = gs.fit(X_train, y_train)
            best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

        # Update best_score_param_estimators
        return best_score_param_estimators

    def model_selection(self, best_score_param_estimators, X_test):
        best_score_param_estimators = sorted(best_score_param_estimators, key=lambda x: x[0], reverse=True)

        # Print best_score_param_estimators
        for rank in range(len(best_score_param_estimators)):
            best_score, best_params, best_estimator = best_score_param_estimators[rank]

            print('Top', str(rank + 1))
            print('%-15s' % 'best_score:', best_score)
            print('%-15s' % 'best_estimator:'.format(20), type(best_estimator.named_steps['estimator']))
            print('%-15s' % 'best_params:'.format(20), best_params, end='\n\n')

        # Get the best estimator
        best_estimator = best_score_param_estimators[0][2]

        y_pred = best_estimator.predict(X_test)



    def feature_importance(self):
        return False

