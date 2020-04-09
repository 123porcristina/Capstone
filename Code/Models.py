# !pip install xgboost
# conda install -c conda-forge xgboost
# conda install -c conda-forge imbalanced-learn
# conda install -c glemaitre imbalanced-learn

import warnings
warnings.filterwarnings("ignore")

# General
from time import time
import pandas as pd
import seaborn as sns
import numpy as np
from time import time
from collections import Counter
from IPython.display import Image
from pathlib import Path
import pandas as pd
import pandas_profiling as pp
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import numpy as np
from catboost import CatBoostClassifier, cv, Pool
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# For encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# For feature importance:
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

# Estimators
from catboost import CatBoostClassifier, cv, Pool
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# Resampling
import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

# metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from catboost.utils import get_confusion_matrix




class ModelAccidents():
    def __init__(self, df):
        self.crashes = df  # get df from the file preprocessing.py

    def split_data(self, df, target):
        X = df.drop([target], axis=1)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
        categorical_features_indices = np.where(X.dtypes != np.float)[0]
        return X, y, X_train, X_test, y_train, y_test, categorical_features_indices

    def feature_importance(self, X, y):
        col = X.columns
        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

        forest.fit(X, y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

        colNew = []

        for f in range(X.shape[1]):
            ind = indices[f]
            colNew.append(col[ind])

        # Plot the feature importances of the forest
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.barh(range(X.shape[1]), importances[indices], tick_label=colNew, color="r", yerr=std[indices],
                 align="center")
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

    @staticmethod
    def feature_importance_lr(X, y):
        # define the model
        model = LinearRegression()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.coef_
        print(np.argsort(importance))
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()
        print(list(X.columns))

    @staticmethod
    def oversampling(X_train, y_train):
        sm = SMOTE(random_state=12)
        X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
        return X_train_res, y_train_res

    @staticmethod
    def oversampling_cat(X_train, y_train):
        over_sampler = RandomOverSampler()
        X_train_res, y_train_res = over_sampler.fit_sample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_train_res))
        return X_train_res, y_train_res

    def model_catboost(self, X, y, X_train, y_train, X_test, y_test, categorical_features_indices, target, file):

        # Adicione esto: inicio
        train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
        validate_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)
        # fin

        #         model=CatBoostClassifier(loss_function='MultiClass',use_best_model=True, random_seed=42)#, class_weights=[1,2,3,4,5,6,7,8,9,10,11])
        model = CatBoostClassifier(loss_function='MultiClass', eval_metric='TotalF1', use_best_model=True,
                                   random_seed=42,
                                   leaf_estimation_method='Newton')

        model.fit(train_pool, eval_set=validate_pool, use_best_model=True, verbose=50, plot=True,
                  early_stopping_rounds=100)

        # cross-validation
        cv_params = model.get_params()
        cv_data = cv(Pool(X, y, cat_features=categorical_features_indices), cv_params, fold_count=10, plot=True)
        print('Precise validation accuracy score: {}'.format(np.max(cv_data)))  # ['TotalF1']
        # fin

        print("PRIMER prediccion")
        print();
        print(model)
        # make predictions
        expected_y = y_test
        predicted_y = model.predict(X_test)
        # summarize the fit of the model
        print();
        print(metrics.classification_report(expected_y, predicted_y))
        print();
        print(metrics.confusion_matrix(expected_y, predicted_y))

        print("SEGUNDO prediccion")
        print(model.best_iteration_, model.best_score_)
        print(model.evals_result_['validation_1']['MultiClass'][-10:])

        # prediction
        pred = model.predict(X_test)
        print("PREDICT")
        print(pred)

        print("print dataframe predictions:")
        cm = pd.DataFrame()
        #         cm['DAMAGE'] = y_test
        cm[target] = y_test
        cm['Predict'] = model.predict(X_test)
        print(cm)

        print("SCORES")
        print(model.score(X_test, y_test))
        cm.to_csv(file)  # , index=False)
        #         cm.to_csv("catboost_prediction.csv")#, index=False)

        # confusion matrix
        print("confusion matrix:")
        #         conf_mat = get_confusion_matrix(model, Pool(X_train, y_train, cat_features=categorical_features_indices))
        conf_mat = get_confusion_matrix(model, Pool(X_test, y_test, cat_features=categorical_features_indices))
        print(conf_mat)

        return model, cv_data

    def classifiers(self, X_train, X_test, y_train, y_test, df, target):
        t0 = time()
        param_grids = {}
        best_score_param_estimators = []
        classifiers = [
            LogisticRegression(random_state=0),
            KNeighborsClassifier(3),
            DecisionTreeClassifier(class_weight='balanced'),
            RandomForestClassifier(class_weight='balanced'),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            XGBClassifier(),
            #             SVC(kernel="rbf", C=0.025, probability=True),
            #             NuSVC(probability=True),
        ]
        #         for classifier in classifiers:
        #             pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        #             pipe.fit(X_train, y_train)
        #             print(classifier)
        #             print("model score: %.3f" % pipe.score(X_test, y_test))
        # #             print(classification_report(y_test, y_prediction))

        # For each classifier
        for classifier in classifiers:
            gs = GridSearchCV(
                estimator=Pipeline([('StandardScaler', StandardScaler()), ('classifier', classifier)]),
                param_grid=param_grids,  # [name],
                scoring='f1_micro',  # 'accuracy',
                n_jobs=1,
                iid=False,
                cv=StratifiedKFold(n_splits=5,
                                   shuffle=True,
                                   random_state=0))
            # Fit the pipeline
            gs = gs.fit(X_train, y_train)
            print('1')

            best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

        print("done in %0.3fs" % (time() - t0))  # returns time it takes to run
        return best_score_param_estimators

    def classifiers3(self, X_train, y_train):  # X_validation, y_validation, df, target):

        t0 = time()

        clfs = {  # 'lr': LogisticRegression(random_state=0),
            #                 'mlp': MLPClassifier(random_state=0),
            #                 'dt': DecisionTreeClassifier(random_state=0),
            #                 'rf': RandomForestClassifier(random_state=0),
            #                 'xgb': XGBClassifier(seed=0),
            'svc': SVC(random_state=0),
            'knn': KNeighborsClassifier(),
            'gnb': GaussianNB()}

        pipe_clfs = {}

        for name, clf in clfs.items():
            # Implement me
            pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()),
                                        ('clf', clf)])
        print(pipe_clfs['gnb'])

        param_grids = {}

        ##LR
        #         C_range = [10 ** i for i in range(-4, 5)]

        #         param_grid = [{'clf__multi_class': ['ovr'],
        #                        'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        #                        'clf__C': C_range},

        #                       {'clf__multi_class': ['multinomial'],
        #                        'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        #                        'clf__C': C_range}]

        #         param_grids['lr'] = param_grid

        #         ###MLP
        #         param_grid = [{'clf__hidden_layer_sizes': [10, 100],
        #                'clf__activation': ['identity', 'logistic', 'tanh', 'relu']}]

        #         param_grids['mlp'] = param_grid

        #         ##DT
        #         param_grid = [{'clf__min_samples_split': [2, 10, 30],
        #                'clf__min_samples_leaf': [1, 10, 30]}]

        #         param_grids['dt'] = param_grid

        #         ##RF
        #         param_grid = [{'clf__n_estimators': [10, 100],
        #                'clf__min_samples_split': [2, 10, 30],
        #                'clf__min_samples_leaf': [1, 10, 30]}]

        #         param_grids['rf'] = param_grid

        #         ##XGBOOST
        #         param_grid = [{'clf__eta': [10 ** i for i in range(-4, 1)],
        #                'clf__gamma': [0, 10, 100],
        #                'clf__lambda': [10 ** i for i in range(-4, 5)]}]

        #         param_grids['xgb'] = param_grid

        ##SVC
        param_grid = [{'clf__C': [10 ** i for i in range(-4, 5)],
                       'clf__gamma': ['auto', 'scale']}]

        param_grids['svc'] = param_grid

        ##KNN
        param_grid = [{'clf__n_neighbors': list(range(1, 11))}]

        param_grids['knn'] = param_grid

        ##GNB
        param_grid = [{'clf__var_smoothing': [10 ** i for i in range(-10, -7)]}]

        param_grids['gnb'] = param_grid

        print(param_grids)

        ####Hyperparameter tuning
        # The list of [best_score_, best_params_, best_estimator_]
        best_score_param_estimators = []

        # For each classifier
        for name in pipe_clfs.keys():
            # GridSearchCV
            gs = GridSearchCV(estimator=pipe_clfs[name],
                              param_grid=param_grids[name],
                              scoring='accuracy',
                              n_jobs=1,
                              iid=False,
                              cv=StratifiedKFold(n_splits=5,
                                                 shuffle=True,
                                                 random_state=0))
            # Fit the pipeline
            gs = gs.fit(X_train, y_train)
            print('1')

            # Update best_score_param_estimators
            best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

        print("done in %0.3fs" % (time() - t0))  # returns time it takes to run
        return best_score_param_estimators

    def model_selection(self, best_score_param_estimators, clf):
        # Sort best_score_param_estimators in descending order of the best_score_
        best_score_param_estimators = sorted(best_score_param_estimators, key=lambda x: x[0], reverse=True)

        # Print best_score_param_estimators
        for rank in range(len(best_score_param_estimators)):
            best_score, best_params, best_estimator = best_score_param_estimators[rank]

            print('Top', str(rank + 1))
            print('%-15s' % 'best_score:', best_score)
            print('%-15s' % 'best_estimator:'.format(20), type(best_estimator.named_steps[clf]))
            print('%-15s' % 'best_estimator:'.format(20), type(best_estimator.named_steps['']))
            print('%-15s' % 'best_params:'.format(20), best_params, end='\n\n')

        # Get the best estimator
        best_estimator = best_score_param_estimators[0][2]
        return best_estimator

    def predict(self, best_estimator, X_test, y_test, le):

        # Predict the target value using the best estimator
        print("Predicting TARGET on the test set")
        t0 = time()
        expected_y = y_test
        y_pred = best_estimator.predict(X_test)

        # gets the original target
        y_pred = le.inverse_transform(y_pred)
        expected_y = le.inverse_transform(expected_y)

        # returns time it takes to run
        print("done in %0.3fs" % (time() - t0), end='\n')
        print(classification_report(expected_y, y_pred), end='\n')
        print(confusion_matrix(expected_y, y_pred), end='\n')
