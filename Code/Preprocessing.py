# pip install pandas-profiling
# pip install --force-reinstall package_with_metadata_issue

# General
import pandas as pd
from pathlib import Path

# EDA
from pandas_profiling import ProfileReport

# For encoding
from sklearn.preprocessing import LabelEncoder


# types of variables and descriptive statistics
class ReadData:
    def __init__(self, path):
        self.path = path

    def read_complete_data(self):
        """Read complete csv after join - this file was previouly downloaded"""
        """for a faster reading, read the final file with the joins"""
        df = pd.read_csv(str(Path(__file__).parents[1]) + '/Data/' + 'Merged_New_Variables_Only.csv')#, dtype='unicode')

        df = df.drop(['CRASH_RECORD_ID', 'RD_NO', 'CRASH_DATE', 'LATITUDE', 'LONGITUDE', 'PRIM_CONTRIBUTORY_CAUSE'],
                     axis=1)  # 'CRASH_DATE_x', 'DATE_POLICE_NOTIFIED', 'HIT_AND_RUN_I', 'RD_NO_y',
        return df

    def get_value_user(self, df):
        # ask to the user to enter which target wants to evaluate
        while True:
            try:
                target = input("Enter TARGET to evaluate: ")
            except ValueError:
                print("Please, enter a value")
                continue
            if target == 'DAMAGE':
                df = df.drop(['CRASH_TYPE', 'Most_Severe_Injury_New'], axis=1)
                break
            elif target == 'CRASH_TYPE':
                df = df.drop(['DAMAGE', 'Most_Severe_Injury_New'], axis=1)
                break
            elif target == 'Most_Severe_Injury_New':
                df = df.drop(['CRASH_TYPE', 'CRASH_TYPE'], axis=1)
                break
            else:
                print("Invalid! ", target)
                continue
        return df, target

    def convert_datetime(self, df):
        df.CRASH_DATE = pd.to_datetime(df.CRASH_DATE)
        return df

    def convert_categorical(self, df):
        cols = df.select_dtypes(exclude=['int64', 'float64', 'datetime']).columns.to_list()
        df[cols] = df[cols].astype('category')
        return df

    def identify_features(self, df):
        """Identify categorical and numerical features"""
        cols = df.select_dtypes(exclude=['int64', 'float64', 'datetime']).columns.to_list()
        df[cols] = df[cols].astype('category')

        numerical_index = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_index = df.select_dtypes(include=['object', 'bool', 'category']).columns
        return numerical_index, categorical_index

    def data_imputation(self, df):
        """Fill in missing values"""
        # categorical
        df["Traffic_Control_New"].fillna((df["Traffic_Control_New"].mode()[0]), inplace=True)
        df["Road_Surface_New"].fillna((df["Road_Surface_New"].mode()[0]), inplace=True)
        df["SEX2"].fillna((df["SEX2"].mode()[0]), inplace=True)
        df['Weather_New'].fillna((df.groupby(['Road_Surface_New'])['Weather_New'].transform(lambda x: x.mode()[0])),
                                 inplace=True)
        # numerical
        df["Posted_Speed_New"].fillna((df["Posted_Speed_New"].median()), inplace=True)
        df["BAC2"].fillna((df["BAC2"].median()), inplace=True)
        # df["AGE2"].fillna((df.groupby('SEX2')["AGE2"].transform("median")), inplace=True)
        df['AGE2'] = df.groupby(["SEX2"])['AGE2'].transform(lambda x: x.fillna(x.median()))
        return df

    def factorize_categ(self, df, target):
        """Ordinal Encoding just for target"""
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        # print(le.classes_)
        return df, le

    def calc_smooth_mean(self, df, by, on, m):
        """target encoding techique using smoothing. It is used to encode all the variables different from the target"""
        # Compute the global mean
        mean = df[on].mean()
        # Compute the number of values and the mean of each group
        agg = df.groupby(by)[on].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        # Compute the "smoothed" means
        smooth = (counts * means + m * mean) / (counts + m)
        # Replace each value by the according smoothed mean
        return df[by].map(smooth)

    def describe_data(self, df):
        return ProfileReport(df, title='Pandas Profiling Report', html={'style': {'full_width': True}})

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
