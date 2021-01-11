"""
python  clean.py
python  clean.py  profile     #### Data Profile
python  clean.py  train_test_split
"""
import pandas as pd, numpy as np
import os
import pandas_profiling
from sklearn.model_selection import train_test_split

#######################################################################################


##### Load from samples   ##################
df = pd.read_csv('raw/transfusion.csv', nrows= 284)
print(df.head(5).T)
print(df.tail(5).T)
print(df.dtypes)

#######################################################################################
"""
Put manually column by data type :
"""

colid = "Id"

coly = "whether_they_donated_blood"  # "PassengerId"

colcat = []

colnum = ["Id","Recency","Frequency","Monetary","Time"]

colsX = colcat + colnum

print('coly', coly)
print('colsX', colsX)


#######################################################################################
#######################################################################################
def profile():
    os.makedirs("profile/", exist_ok=True)
    for x in colcat:
        df[x] = df[x].factorize()[0]

    ##### Pandas Profile   ###################################
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(output_file="profile/raw_report.html")


print("profile/raw_report.html")


#######################################################################################
#######################################################################################
def create_features(df):
    return df


def train_test_split():
    os.makedirs("train/", exist_ok=True)
    os.makedirs("test/", exist_ok=True)
    df1 = df
    df1_train = pd.DataFrame()
    df1_test = pd.DataFrame()
    icol = int(0.8 * len(df1))
    df1[colsX].iloc[:icol, :].to_parquet("train/features.parquet")
    df1[[coly]].iloc[:icol, :].to_parquet("train/target.parquet")
    df1_train[colsX] = df[colsX].iloc[:icol, :]
    df1_train[[coly]] = df[[coly]].iloc[:icol, :]
    df1_train.to_csv("train/train.csv")


    df1[colsX].iloc[icol:, :].to_parquet("test/features.parquet")
    df1[[coly]].iloc[icol:, :].to_parquet("test/target.parquet")
    df1_test[colsX] = df[colsX].iloc[icol:, :]
    df1_test[[coly]] = df[[coly]].iloc[icol:, :]
    df1_test.to_csv("test/test.csv")
########################################################################################
"""
python  clean.py
python  clean.py  profile
python  clean.py  to_train
python  clean.py  to_test
"""
if __name__ == "__main__":
    import fire

    fire.Fire()
