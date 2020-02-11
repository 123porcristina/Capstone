from pathlib import Path
import pandas as pd


class ReadData:
    def __init__(self, data_crashes, data_people, data_vehicle):
        self.crashes = data_crashes
        self.people = data_people
        self.vehicle = data_vehicle

    def read_dataset(self):
        self.crashes = pd.read_csv(str(Path(__file__).parents[1]) + '/Data/' + self.crashes)
        self.people  = pd.read_csv(str(Path(__file__).parents[1]) + '/Data/' + self.people)
        self.vehicle = pd.read_csv(str(Path(__file__).parents[1]) + '/Data/' + self.vehicle)
        df = self.join_data()
        return df#self.crashes, self.people, self.vehicle

    def join_data(self):
        # primary key RD_NO

        return False

#     def drop_cols(self):
#         return False
#
#
# class CleanData:
#     def __init__(self):
#         return False
#
#     # Hot-Encoding: SAFETY_EQUIPMENT, DRIVER_ACTION, DRIVER_VISION, PHYSICAL_CONDITION, PERSON_TYPE
#
#
# class EDA:
#     def __init__(self):
#         return False
#
#     def check(self):
#         return False

a = ReadData("Traffic_Crashes_-_Crashes (1).csv", "Traffic_Crashes_-_People.csv", "Traffic_Crashes_-_Vehicles.csv")
a1 = a.read_dataset()
print(a1.head(10))

