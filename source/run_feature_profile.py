# -*- coding: utf-8 -*- 
import gc
import os
import logging
from datetime import datetime
import warnings
import numpy as np
import pandas as pd

import pandas_profiling as pp

###################################################################
path = os.path.abspath(os.getcwd() + "/../")

coly = 'salary'
colid = "jobId"
colcat = ['companyId', 'jobType', 'degree', 'major', 'industry']
colnum = ['yearsExperience', 'milesFromMetropolis']
coltext = []
coldate = []

#### Pandas Profiling for features in train  ######################
df = pd.read_csv(path + f"/data/input/train/features.zip")
dfy = pd.read_csv(path + f"/data/input/train/target_values.zip")
df = df.join(dfy.set_index(colid), on=colid, how="left")

df = df.set_index(colid)
for x in colcat:
    df[x] = df[x].factorize()[0]

profile = df.profile_report(title='Profile Test data')
profile.to_file(output_file=path + "/analysis/00_features_train_report.html")

#### Test dataset  ################################################
df = pd.read_csv(path + f"/data/input/test/features.zip")
df = df.set_index(colid)
for x in colcat:
    df[x] = df[x].factorize()[0]

profile = df.profile_report(title='Profile Test data')
profile.to_file(output_file=path + "/analysis/00_features_test_report.html")
