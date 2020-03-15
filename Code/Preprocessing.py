
# pip install pandas-profiling
# pip install --force-reinstall package_with_metadata_issue
import pandas as pd
import numpy as np
from pathlib import Path
from pandas_profiling import ProfileReport
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



# types of variables and descriptive statistics
class ReadData:
    def __init__(self, data_crashes, data_people, data_vehicle, drop_cols):
        self.crashes = data_crashes
        self.people = data_people
        self.vehicle = data_vehicle
        self.drop_col = drop_cols

    def read_dataset(self):
        """Read all datasets and join them, also read file to drop cols"""
        self.crashes = pd.read_csv(str(Path(__file__).parents[1]) + '/Data/' + self.crashes, dtype='unicode')
        self.people = pd.read_csv(str(Path(__file__).parents[1]) + '/Data/' + self.people, dtype='unicode')
        self.vehicle = pd.read_csv(str(Path(__file__).parents[1]) + '/Data/' + self.vehicle, dtype='unicode')
        self.drop_col = list(pd.read_csv(str(Path(__file__).parents[1]) + '/Data/' + self.drop_col))
        df = self.join_data()
        return df

    def read_complete_data(self):
        """Read complete csv after join - this file was previouly downloaded"""
        return pd.read_csv(str(Path(__file__).parents[1]) + '/Data/' + 'final_accidents.csv', dtype='unicode')

    def join_data(self):
        """Join people, crashes and vehicles dataset"""
        df = self.crashes.merge(self.people, on='CRASH_RECORD_ID').merge(self.vehicle, on=['CRASH_RECORD_ID', 'VEHICLE_ID'])
        df = df.drop(self.drop_col, axis=1)  # Drops columns according to excel
        df = self.drop_non_drivers(df)
        df = self.convert_datetime(df)
        df = df.sort_values(by='CRASH_DATE_x', ascending=True)
        df = self.convert_categorical(df)
        return df

    def drop_non_drivers(self, df):
        """deletes all values different than driver"""
        return df[df.PERSON_TYPE == 'DRIVER']

    def convert_datetime(self, df):
        """convert date to date time"""
        # df.DATE_POLICE_NOTIFIED = pd.to_datetime(df.DATE_POLICE_NOTIFIED)
        df.CRASH_DATE_x = pd.to_datetime(df.CRASH_DATE_x)
        return df

    def convert_categorical(self, df):
        """convert object to categorical to speed training"""
        cols = df.select_dtypes(exclude=['int64', 'float64', 'datetime']).columns.to_list()
        df[cols] = df[cols].astype('category')
        return df

    def describe_data(self, df):
        """Using pandas-profiling to create our: EDA - it takes long time to run"""
        df = df.drop(['CRASH_DATE_x', 'DATE_POLICE_NOTIFIED', 'HIT_AND_RUN_I', 'RD_NO_y', 'RD_NO'], axis=1)
        return ProfileReport(df, title='EDA Report', html={'style': {'full_width': True}})

    def download_preprocessing(self, df):
        return df.to_csv(r'C:/Users/Cristina/Documents/GWU/Capstone/Data/final_accidents2.csv', index=False)

    def identify_features(self, df):
        """identify categorical and num features"""
        num_features = df.select_dtypes(include=['int64', 'float64']).columns
        cat_features = df.select_dtypes(include=['object', 'bool', 'category']).columns
        return num_features, cat_features


    def encode_imputation(self, num_features, cat_features, df):
        "Dealing with missing values and One Hot Encode categorical vbles"
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                                          ('scaler', StandardScaler())])

        cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        col_transform = ColumnTransformer(transformers=[('num', num_transformer, num_features),
                                                        ('cat', cat_transformer, cat_features)])

        col_transform.fit(df)
        df = col_transform.transform(df)
        return col_transform, df




# file1 = "Traffic_Crashes_-_Crashes.csv"
# file2 = "Traffic_Crashes_-_People.csv"
# file3 = "Traffic_Crashes_-_Vehicles.csv"
#
# crash = ReadData(file1, file2, file3, "drop_cols.csv")
#
# df_crashes = crash.read_dataset()
# crash.download_preprocessing(df_crashes)
# print("Download is done!")
# # df_crashes = crash.read_joing()
# profile = crash.describe_data(df_crashes)
# profile.to_file('C:/Users/Cristina/Documents/GWU/Capstone/EDA_accidents_pycharm2.html')
# #