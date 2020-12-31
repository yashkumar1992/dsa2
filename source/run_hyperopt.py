# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:45:07 2020


Input a dict of variables.

pars_dict

pars_dict_range :


Optimize all

obj_fun




"""





import pandas as pd
import numpy as np
#import xgboost as xgb
import lightgbm as lgb
import gc

from skopt.space import Real, Integer
from skopt.utils import use_named_args
import itertools
from sklearn.metrics import roc_auc_score
from skopt import gp_minimize




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, json
from pandas.io.json import json_normalize
import lightgbm as lgb
from sklearn.feature_selection import RFE

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output


https://rbfopt.readthedocs.io/en/latest/rbfopt_aux_problems.html





“RBFOpt: an open-source library for black-box optimization with costly function ...

https://github.com/coin-or/rbfopt

After installation, the easiest way to optimize a function is to use the RbfoptUserBlackBox class to define a black-box, and execute RbfoptAlgorithm on it. This is a minimal example to optimize the 3-dimensional function defined below:

import rbfopt
import numpy as np
def obj_funct(x):
  return x[0]*x[1] - x[2]

bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
                               np.array(['R', 'I', 'R']), obj_funct)
settings = rbfopt.RbfoptSettings(max_evaluations=50)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()
Another possibility is to define your own class derived from RbfoptBlackBox in a separate file, and execute the command-line interface on the file. An example is provided under src/rbfopt/examples, in the file rbfopt_black_box_example.py. This can be executed with:

rbfopt_cl_interface.py src/rbfopt/examples/rbfopt_black_box_example.py







# Feature importance

#lightGBM model fit
gbm = lgb.LGBMRegressor()
gbm.fit(train, target)
gbm.booster_.feature_importance()

# importance of each attribute
fea_imp_ = pd.DataFrame({'cols':train.columns, 'fea_imp':gbm.feature_importances_})
fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)







TRAINING_SIZE = 300000


#Recursive Feature Elimination(RFE)

# create the RFE model and select 10 attributes
rfe = RFE(gbm, 10)
rfe = rfe.fit(train, target)

# summarize the selection of the attributes
print(rfe.support_)

# summarize the ranking of the attributes
fea_rank_ = pd.DataFrame({'cols':train.columns, 'fea_rank':rfe.ranking_})
fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by=['fea_rank'], ascending = True)















TEST_SIZE = 50000

# Load data
train = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TRAINING_SIZE-TEST_SIZE), 
    nrows=TRAINING_SIZE,
    parse_dates=['click_time']
)

val = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TEST_SIZE), 
    nrows=TEST_SIZE,
    parse_dates=['click_time']
)

# Split into X and y
y_train = train['is_attributed']
y_val = val['is_attributed']


# from that dimension (`'log-uniform'` for the learning rate)
space  = [Integer(3, 10, name='max_depth'),
          Integer(6, 30, name='num_leaves'),
          Integer(50, 200, name='min_child_samples'),
          Real(1, 400,  name='scale_pos_weight'),
          Real(0.6, 0.9, name='subsample'),
          Real(0.6, 0.9, name='colsample_bytree')
         ]





res_gp = gp_minimize(objective, space, n_calls=20,
                     random_state=0,n_random_starts=10)

"Best score=%.4f" % res_gp.fun






TRAINING_SIZE = 300000
TEST_SIZE = 50000

# Load data
train = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TRAINING_SIZE-TEST_SIZE), 
    nrows=TRAINING_SIZE,
    parse_dates=['click_time']
)

val = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TEST_SIZE), 
    nrows=TEST_SIZE,
    parse_dates=['click_time']
)

# Split into X and y
y_train = train['is_attributed']
y_val = val['is_attributed']

Specify the parameter space we want to explore.

# from that dimension (`'log-uniform'` for the learning rate)
space  = [Integer(3, 10, name='max_depth'),
          Integer(6, 30, name='num_leaves'),
          Integer(50, 200, name='min_child_samples'),
          Real(1, 400,  name='scale_pos_weight'),
          Real(0.6, 0.9, name='subsample'),
          Real(0.6, 0.9, name='colsample_bytree')
         ]

Below is the fun part. The function gp_minimize requires an objective function and what the function all needs is basically a metric we want to minimize. Of course, we can just use whatever training setup we have been using but just tweak it to return a AUC to minimize..(negative AUC)

def objective(values):
    

    params = {'max_depth': values[0], 
          'num_leaves': values[1], 
          'min_child_samples': values[2], 
          'scale_pos_weight': values[3],
            'subsample': values[4],
            'colsample_bytree': values[5],
             'metric':'auc',
             'nthread': 8,
             'boosting_type': 'gbdt',
             'objective': 'binary',
             'learning_rate':0.15,
             'max_bin': 100,
             'min_child_weight': 0,
             'min_split_gain': 0,
             'subsample_freq': 1}
    

    print('\nNext set of params.....',params)
    
    feature_set = ['app','device','os','channel']
    categorical = ['app','device','os','channel']
    
    
    early_stopping_rounds = 50
    num_boost_round       = 1000
    
        # Fit model on feature_set and calculate validation AUROC
    xgtrain = lgb.Dataset(train[feature_set].values, label=y_train,feature_name=feature_set,
                           categorical_feature=categorical)
    xgvalid = lgb.Dataset(val[feature_set].values, label=y_val,feature_name=feature_set,
                          categorical_feature=categorical)
    
    evals_results = {}
    model_lgb     = lgb.train(params,xgtrain,valid_sets=[xgtrain, xgvalid], 
                              valid_names=['train','valid'], 
                               evals_result=evals_results, 
                               num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=None, feval=None)
    
    auc = -roc_auc_score(y_val, model_lgb.predict(val[model_lgb.feature_name()]))
    
    print('\nAUROC.....',-auc,".....iter.....", model_lgb.current_iteration())
    
    gc.collect()
    
    return  auc





