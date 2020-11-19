# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Run template

"""
import warnings, copy
warnings.filterwarnings('ignore')
import os, sys
import pandas as pd

####################################################################################
from source import util_feature


config_name = None
path_config_model = None
path_model = None
path_data_train = None
path_data_test = None
path_output_pred = None
n_sample = 10
model_name = None



###################################################################################
########## Preprocess #############################################################
def preprocess():
   from source import run_preprocess
   run_preprocess.run_preprocess(model_name='cardif_lightgbm',
                              path_data=f'data/input/{data_name}/train/',
                              path_output=f'data/output/{data_name}/a01_{model_name}/',
                              path_config_model="source/config_model.py",
                              n_sample=5000,
                              mode='run_preprocess'
                              )




###################################################################################
########## Train ##################################################################
def train():
    from source import run_train

    print(config_name, path_data_train)

    run_train.run_train(config_model_name =  config_name,
                        path_data         =  path_data_train,
                        path_output       =  path_model,
                        path_config_model =  path_config_model , n_sample = n_sample)


###################################################################################
######### Check model #############################################################
def check():
    try :
        #### Load model
        from source.util_feature import load
        from source.models import model_sklearn as modelx
        import sys
        from source import models
        sys.modules['models'] = models

        dir_model    = path_model
        modelx.model = load( dir_model + "/model/model.pkl" )
        stats        = load( dir_model + "/model/info.pkl" )
        colsX        = load( dir_model + "/model/colsX.pkl"   )
        coly         = load( dir_model + "/model/coly.pkl"   )
        print(stats)
        print(modelx.model.model)

        ### Metrics on test data
        stats['metrics_test']

        #### Loading training data  #######################################################
        dfX     = pd.read_csv(dir_model + "/check/dfX.csv")  #to load csv
        #dfX = pd.read_parquet(dir_model + "/check/dfX.parquet")    #to load parquet
        dfy     = dfX[coly]
        colused = colsX

        dfXtest = pd.read_csv(dir_model + "/check/dfXtest.csv")    #to load csv
        #dfXtest = pd.read_parquet(dir_model + "/check/dfXtest.parquet"    #to load parquet
        dfytest = dfXtest[coly]
        print(dfX.shape,  dfXtest.shape )


        #### Feature importance on training data
        lgb_featimpt_train,_ = util_feature.feature_importance_perm(modelx, dfX[colused], dfy, colused, n_repeats=1,
                                                                    scoring='accuracy' )

        print(lgb_featimpt_train)
    except :
        pass
    #! python source/run_inference.py  run_predict  --config_model_name  LGBMRegressor  --n_sample 1000   --path_model /data/output/a01_lightgbm_huber/    --path_output /data/output/pred_a01_lightgbm_huber/    --path_data /data/input/train/



########################################################################################
####### Inference ######################################################################
def predict():
    from source import run_inference
    run_inference.run_predict(model_name,
                              path_model  = path_model,
                              path_data   = path_data_test,
                              path_output = path_output_pred,
                              n_sample    = n_sample)


def run_all():
    train()
    check()
    predict()


