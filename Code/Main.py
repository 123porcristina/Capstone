from Code import Preprocessing as preprocessing
from Code import Models as Model
from collections import Counter
import numpy as np


def main():
    path = 'C:/Users/Cristina/Documents/GWU/Capstone/Data/'  # from where to read
    path_download = 'C:/Users/Cristina/Documents/GWU/Capstone/EDA_accidents_pycharm2.html'  # where to download

    accidents = preprocessing.ReadData(path=path)
    """Read preprocessed dataset and fill in missing values"""
    df = accidents.read_complete_data()
    df = accidents.data_imputation(df)

    """Generates EDA"""
    # it is commented because it takes a long time to run- uncomment if needed
    # profile = accidents.describe_data(df)
    # profile.to_file(path_download)

    """User action required: Enter target to evaluate"""
    df, target = accidents.get_value_user(df)

    # """Encoding Features"""
    # # ENCODE VARIABLES (FACTORIZE: ORDINAL (TARGET))
    # df_encoded, le = accidents.factorize_categ(df, target=target)
    #
    # # ENCODE TARGET TECHNIQUE USING SMOOTH: OTHER FEATURES
    # # df_encoded['PRIM_CONTRIBUTORY_CAUSE'] = accidents.calc_smooth_mean(df_encoded, by='PRIM_CONTRIBUTORY_CAUSE',
    # #                                                                    on=target, m=10)
    # df_encoded['FIRST_CRASH_TYPE'] = accidents.calc_smooth_mean(df_encoded, by='FIRST_CRASH_TYPE', on=target, m=10)
    # df_encoded['Contributory_Cause_New'] = accidents.calc_smooth_mean(df_encoded, by='Contributory_Cause_New',
    #                                                                   on=target, m=10)
    # df_encoded['Traffic_Control_New'] = accidents.calc_smooth_mean(df_encoded, by='Traffic_Control_New', on=target,
    #                                                                m=10)
    # df_encoded['Weather_New'] = accidents.calc_smooth_mean(df_encoded, by='Weather_New', on=target, m=10)
    # df_encoded['Road_Surface_New'] = accidents.calc_smooth_mean(df_encoded, by='Road_Surface_New', on=target, m=10)
    # df_encoded['SEX2'] = accidents.calc_smooth_mean(df_encoded, by='SEX2', on=target, m=10)
    # print("Encoded DONE!")

    """Instance class to model accidents"""
    model = Model.ModelAccidents(df=df)

    # """SHALLOW MODELS EXCEPT CATBOOST"""
    # # ENCODED: SPLIT IN TRAINING AND TEST
    # X, y, X_train, X_validation, y_train, y_validation, categorical_features_indices = model.split_data(df_encoded,
    #                                                                                                     target=target)
    # # Feature Importance using logistic regression and random forest
    # print(model.feature_importance_lr(X, y))
    # print("CReating feature importance 2")
    # print(model.feature_importance(X, y))
    #
    # # Oversampling
    # X_train_res, y_train_res = model.oversampling(X_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_train_res))
    #
    # # SHALLOW CLASSIFIERS-OVERSAMPLED - ENCODED
    # X, y, X_train, X_validation, y_train, y_validation, categorical_features_indices = model.split_data(df_encoded,
    #                                                                                                     target=target)
    # # Executes several classifiers
    # best_score_param_estimators = model.classifiers(X_train_res, X_validation, y_train_res, y_validation, df_encoded,
    #                                                     target=target)
    # # Gets best classifier to be used for prediction
    # best_estimator = model.model_selection(best_score_param_estimators, clf='classifier')
    #
    # """Predictions for the best model"""
    # print(model.predict(best_estimator, X_validation, y_validation, le))


    """********CATBOOST*********"""

    # NOT ENCODED - NO RESAMPLED - CATBOOST

    X, y, X_train, X_test, y_train, y_test, categorical_features_indices = model.split_data(df, target=target)
    # model_cb, cv_data = model.model_catboost(X, y, X_train, y_train, X_test, y_test, categorical_features_indices,
    #                                          target=target, file="catboost_accidents_predictionNOsampled.csv")
    # print('the best cv accuracy is :{}'.format(np.max(cv_data)))

    # NOT ENCODED - OVERSAMPLED - CATBOOST
    X_train_res, y_train_res = model.oversampling_cat(X_train, y_train, categorical_features_indices)
    print("Starting catboost.....")
    model_cb = model.model_catboost(X_train_res, y_train_res, X_test, y_test, categorical_features_indices,
                                    target=target, file="catboost_accidents_predictionSampled.csv")

    # Get feature importance using catboost
    model_cb.get_feature_importance(prettified=True)


if __name__ == '__main__':
    main()
