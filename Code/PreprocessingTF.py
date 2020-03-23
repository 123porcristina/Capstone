#Import Packages
import pandas as pd
import numpy as np

#-------------------------------------------------  CRASH DATA ---------------------------------------------------------
crash= pd.read_csv("../Data/Traffic_Crashes_-_Crashes.csv")

#Drop Columns
crash=crash[['CRASH_RECORD_ID','RD_NO','CRASH_DATE', 'POSTED_SPEED_LIMIT', 'TRAFFIC_CONTROL_DEVICE','WEATHER_CONDITION',
             'LIGHTING_CONDITION' ,'ROADWAY_SURFACE_COND','CRASH_TYPE','FIRST_CRASH_TYPE',
             'DAMAGE', 'NUM_UNITS', 'MOST_SEVERE_INJURY',
             'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH',
             'WORK_ZONE_I', 'LATITUDE', 'LONGITUDE']]
print(crash.info())

#Limit Crashes to 2 Units Crashes (88% of crashes)
print(crash.NUM_UNITS.value_counts(normalize=True, dropna=False))
crash=crash[crash['NUM_UNITS']==2]
print(crash.info())


#Drop Speed Limits below 15 and above 70 and replace with np.nan, leaves 5% nan
print(crash.POSTED_SPEED_LIMIT.value_counts(normalize=True, dropna=False))
def new_speed_limit(row):
    val = row['POSTED_SPEED_LIMIT']
    if row['POSTED_SPEED_LIMIT'] < 15:
        return np.nan
    if row['POSTED_SPEED_LIMIT'] > 70:
        return np.nan

    else:
        return val
crash['Posted_Speed_New'] = crash.apply(new_speed_limit, axis=1)
print(crash.Posted_Speed_New.value_counts(normalize=True, dropna=False))



#Make Traffic Controls 3 categories instead of 18, changed unknown to np.nan, Nan 3%
print(crash.TRAFFIC_CONTROL_DEVICE.value_counts(normalize=True, dropna=False))
def new_traffic_device(row):
    if row['TRAFFIC_CONTROL_DEVICE'] == 'NO CONTROLS':
        return "None"
    if row['TRAFFIC_CONTROL_DEVICE'] == 'TRAFFIC SIGNAL':
        return "TrafficSignal"
    if row['TRAFFIC_CONTROL_DEVICE'] == 'UNKNOWN':
        return np.nan
    else:
        return "OtherControl"
crash['Traffic_Control_New'] = crash.apply(new_traffic_device, axis=1)
print(crash.Traffic_Control_New.value_counts(normalize=True, dropna=False))

#Made Weather Condtion 4 categories instead of 11, Changed Unknown to np.nan, Nan 4%
print(crash.WEATHER_CONDITION.value_counts(normalize=True, dropna=False))
def new_weather_condition(row):
    if row['WEATHER_CONDITION'] == 'CLEAR':
        return "Clear"
    if row['WEATHER_CONDITION'] == 'RAIN':
        return "Rain"
    if row['WEATHER_CONDITION'] == 'SNOW':
        return "Snow"
    if row['WEATHER_CONDITION'] == 'CLOUDY/OVERCAST':
        return "Cloudy"
    if row['WEATHER_CONDITION'] == 'UNKNOWN':
        return np.nan
    else:
        return "OtherForecast"
crash['Weather_New'] = crash.apply(new_weather_condition, axis=1)
print(crash.Weather_New.value_counts(normalize=True, dropna=False))



#Changed Unknown Lighting Conditions np.nan, Nan 4%
print(crash.LIGHTING_CONDITION.value_counts(normalize=True, dropna=False))
crash['LIGHTING_CONDITION']=crash['LIGHTING_CONDITION'].replace('UNKNOWN', np.nan)
print(crash.LIGHTING_CONDITION.value_counts(normalize=True, dropna=False))


#Modified Road Condtions Categories, Changed Unknown to Nan, Nan 7%
print(crash.ROADWAY_SURFACE_COND.value_counts(normalize=True, dropna=False))
def new_road_surface(row):
    if row['ROADWAY_SURFACE_COND'] == 'DRY':
        return "Dry"
    if row['ROADWAY_SURFACE_COND'] == 'WET':
        return "Wet"
    if row['ROADWAY_SURFACE_COND'] == 'SNOW OR SLUSH':
        return "Snow"
    if row['ROADWAY_SURFACE_COND'] == 'ICE':
        return "Snow"
    if row['ROADWAY_SURFACE_COND'] == 'UNKNOWN':
        return np.nan
    else:
        return "OtherCondition"
crash['Road_Surface_New'] = crash.apply(new_road_surface, axis=1)
print(crash.Road_Surface_New.value_counts(normalize=True, dropna=False))




# No Changes To Crash Type
print(crash.CRASH_TYPE.value_counts(normalize=True, dropna=False))

#Drop Pedestrian, Pedalcylist, Animal, Train crashes, 4% of data
print(crash.FIRST_CRASH_TYPE.value_counts(normalize=True, dropna=False))
crash = crash.drop(crash[(crash['FIRST_CRASH_TYPE'] == 'PEDESTRIAN') | (crash['FIRST_CRASH_TYPE'] == 'PEDALCYCLIST') |
                (crash['FIRST_CRASH_TYPE'] == 'ANIMAL') | (crash['FIRST_CRASH_TYPE'] == 'TRAIN') ].index)
print(crash.FIRST_CRASH_TYPE.value_counts(normalize=True, dropna=False))


# No Changes to Damage
print(crash.DAMAGE.value_counts(normalize=True, dropna=False))


# Non-Incap with Reported and Combine Incap W/ Fatal
print(crash.MOST_SEVERE_INJURY.value_counts(normalize=True, dropna=False))
def new_most_severe_injury(row):
    val = row['MOST_SEVERE_INJURY']
    if row['MOST_SEVERE_INJURY'] == 'REPORTED, NOT EVIDENT':
        return "NONINCAPACITATING INJURY"
    if row['MOST_SEVERE_INJURY'] == 'INCAPACITATING INJURY':
        return "INCAPACITATING OR FATAL"
    if row['MOST_SEVERE_INJURY'] == 'FATAL':
        return "INCAPACITATING OR FATAL"
    else:
        return val
crash['Most_Severe_Injury_New'] = crash.apply(new_most_severe_injury, axis=1)
print(crash.Most_Severe_Injury_New.value_counts(normalize=True, dropna=False))

print(crash.WORK_ZONE_I.value_counts(normalize=True, dropna=False))
print(crash.CRASH_HOUR.value_counts(normalize=True, dropna=False))
print(crash.CRASH_DAY_OF_WEEK.value_counts(normalize=True, dropna=False))
print(crash.CRASH_MONTH.value_counts(normalize=True, dropna=False))

crash.to_csv('../Data/Crash_Modified.csv')

#--------------------------------------- Person Data -------------------------------------------------------------------

person = pd.read_csv("../Data/Traffic_Crashes_-_People.csv")
person = person[['CRASH_RECORD_ID', 'PERSON_ID','PERSON_TYPE','SEX',
                 'BAC_RESULT VALUE']]

#Fill Replace Nan Values with 0 for BAC
print(person['BAC_RESULT VALUE'].value_counts(normalize=True, dropna=False))
person['BAC_RESULT VALUE'] = person['BAC_RESULT VALUE'].fillna(0)


# Combine X and U into one category for sex
print(person.SEX.value_counts(normalize=True, dropna=False))
def new_sex(row):
    val = row['SEX']
    if row['SEX'] == 'X':
        return "O"
    if row['SEX'] == 'U':
        return "O"
    else:
        return val
person['Sex_New'] = person.apply(new_sex, axis=1)
print(person.Sex_New.value_counts(normalize=True, dropna=False))

#Replace Null Values w/ M for Sex
person['Sex_New'] = person['Sex_New'].fillna("M")
print(person.Sex_New.value_counts(normalize=True, dropna=False))

#Limit Dataset to Drivers
print(person.info())
print(person.PERSON_TYPE.value_counts(normalize=True, dropna=False))
person=person[person['PERSON_TYPE']=='DRIVER']
person = person[['CRASH_RECORD_ID', 'PERSON_ID', 'Sex_New',
                 'BAC_RESULT VALUE']]


#sort by crash_record_id and then reset Index
person= person.sort_values(by=['CRASH_RECORD_ID'])
person=person.reset_index()

#Create New Person Record To Help With Pivot
new_person_id = np.zeros(len(person))
print(new_person_id.shape)
for i in range(len(new_person_id)):
    if i == 0:
        new_person_id[i] = 1
    else:
        if person.CRASH_RECORD_ID[i] != person.CRASH_RECORD_ID[i-1]:
            new_person_id[i] = 1
        if person['CRASH_RECORD_ID'][i] == person['CRASH_RECORD_ID'][i-1]:
            new_person_id[i] = new_person_id[i-1]+1
person["New_Person_ID"] = new_person_id

#Now Change to String Values
print(person.New_Person_ID.value_counts())
def person2(row):
    if row['New_Person_ID'] == 1:
        return "one"
    if row['New_Person_ID'] == 2:
        return "two"
    if row['New_Person_ID'] == 3:
        return "three"
    if row['New_Person_ID'] == 4:
        return "four"
    if row['New_Person_ID'] == 5:
        return "five"
    if row['New_Person_ID'] == 6:
        return "six"
    if row['New_Person_ID'] == 7:
        return "seven"
    if row['New_Person_ID'] == 8:
        return "eight"
    if row['New_Person_ID'] == 9:
        return "nine"
    if row['New_Person_ID'] == 10:
        return "ten"
    else:
        return 'check'
person['Person2'] = person.apply(person2, axis=1)
print(person.head())
print(person.Person2.value_counts())

#Export to CSV
person.to_csv('../Data/Person_Modified1.csv')

#------------------------------------------- Bring Everthing to One Level -----------------------------
person=pd.read_csv('../Data/Person_Modified1.csv')
print(person.info())
person = person[['CRASH_RECORD_ID', 'Person2','Sex_New', 'BAC_RESULT VALUE']]

#Create Multilevel Index
person = person.set_index(['CRASH_RECORD_ID', 'Person2'])
print(person.head(10))
print(person.info())

# Use Unstack to create Pivot
person2 = person.unstack()
print(person2.head())
print(person2.info())

#Drop Columns for Persons beyond Person one and two
person2 = person2.drop(columns=[('Sex_New', 'three'), ('Sex_New', 'four'), ('Sex_New', 'five'), ('Sex_New', 'six'),
                                ('Sex_New', 'seven'), ('Sex_New', 'eight'), ('Sex_New', 'nine'), ('Sex_New', 'ten')])

person2 = person2.drop(columns=[('BAC_RESULT VALUE', 'three'), ('BAC_RESULT VALUE', 'four'),
                                ('BAC_RESULT VALUE', 'five'), ('BAC_RESULT VALUE', 'six'),
                                ('BAC_RESULT VALUE', 'seven'), ('BAC_RESULT VALUE', 'eight'),
                                ('BAC_RESULT VALUE', 'nine'), ('BAC_RESULT VALUE', 'ten')])
print(person2.info())

#Check Person Value Counts before creating new sex variable
print(person2[('Sex_New', 'one')].value_counts(normalize=True, dropna=False))
print(person2[('Sex_New', 'two')].value_counts(normalize=True, dropna=False))

#Create New Sex Variable
def new_sex_variable(row):
    if (row[('Sex_New', 'one')] == 'M') & (row[('Sex_New', 'two')] == 'M'):
        return "BothMale"
    if (row[('Sex_New', 'one')] == 'F') & (row[('Sex_New', 'two')] == 'F'):
        return "BothFemale"
    if (row[('Sex_New', 'one')] == 'O') & (row[('Sex_New', 'two')] == 'O'):
        return "BothOther"
    if (row[('Sex_New', 'one')] == 'M') & (row[('Sex_New', 'two')] == 'F'):
        return "MaleFemale"
    if (row[('Sex_New', 'one')] == 'F') & (row[('Sex_New', 'two')] == 'M'):
        return "MaleFemale"
    if (row[('Sex_New', 'one')] == 'M') & (row[('Sex_New', 'two')] == 'O'):
        return "MaleOther"
    if (row[('Sex_New', 'one')] == 'O') & (row[('Sex_New', 'two')] == 'M'):
        return "MaleOther"
    if (row[('Sex_New', 'one')] == 'F') & (row[('Sex_New', 'two')] == 'O'):
        return "FemaleOther"
    if (row[('Sex_New', 'one')] == 'O') & (row[('Sex_New', 'two')] == 'F'):
        return "FemaleOther"
    else:
        return 'check'
person2['SEX2'] = person2.apply(new_sex_variable, axis=1)
print(person2.SEX2.value_counts())


#Create New BAC Variable
def new_bac(row):
    val = row[('BAC_RESULT VALUE', 'one')] + row[('BAC_RESULT VALUE', 'two')]
    return val
person2['BAC2'] = person2.apply(new_bac, axis=1)
print(person2.BAC2.value_counts())
print(person2.info)

#Export to CSV
person2.to_csv('../Data/Person_Modified2.csv')

# ----------------------------------------------- Merge Crash and Person Datasets -------------------------------------------

crash = pd.read_csv('../Data/Crash_Modified.csv', index_col='CRASH_RECORD_ID')
print(crash.info())
print(crash.head())
person = pd.read_csv('../Data/Person_Modified2.csv', index_col='Unnamed: 0')
print(person.info())
print(person.head())

merge = pd.merge(crash,person, how='inner', left_index=True, right_index=True)
print(merge.info())
print(merge.head())

merge = merge.drop(columns=['Unnamed: 0'])

merge2=merge.drop(columns=['POSTED_SPEED_LIMIT','TRAFFIC_CONTROL_DEVICE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
                           'ROADWAY_SURFACE_COND', 'MOST_SEVERE_INJURY', 'WORK_ZONE_I',
                           'Sex_New','Sex_New.1','BAC_RESULT VALUE', 'BAC_RESULT VALUE.1'])

print(merge2.info())

merge.to_csv('../Data/Merged_With_Old_Variables.csv')
merge2.to_csv('../Data/Merged_New_Variables_Only.csv')
