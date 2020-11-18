# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to titatinic dataet
All in one file config

!  python titanic_classifier.py  train
!  python titanic_classifier.py  check
!  python titanic_classifier.py  predict



"""
"""
%load_ext autoreload
%autoreload
%matplotlib inline
%config IPCompleter.greedy=True
"""
import warnings, copy
warnings.filterwarnings('ignore')
import os, sys
import pandas as pd

############################################################################
from source import util_feature



###### Path ################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)



##### Params#######################################################################
def titanic_lightgbm(path_model_out) :
    """
       titanic
    """
    def post_process_fun(y):
        ### After prediction is done
        return  y.astype('int')

    def pre_process_fun(y):
        ### Before the prediction is done
        return  y.astype('int')

    model_dict = {'model_pars': {'config_model_name': 'LGBMClassifier'    ## Class name for model_sklearn.py
        , 'model_path': path_model_out
        , 'model_pars': {'objective': 'binary','learning_rate':0.03,'boosting_type':'gbdt' }  # default
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' :  None ,
                               }
                                 },
      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },

      'data_pars': {
          'cols_input_type' : {
                     "coly"   :   "Survived"
                    ,"colid"  :   "PassengerId"
                    ,"colcat" :   [  "Sex", "Embarked" ]
                    ,"colnum" :   ["Pclass", "Age","SibSp", "Parch","Fare"]
                    ,"coltext" :  ["Name","Ticket"]
                    ,"coldate" :  []
                    ,"colcross" : [ "Name", "Sex", "Ticket","Embarked","Pclass", "Age","SibSp", "Parch","Fare" ]
                   },

          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']
         ,'cols_model':       []  # cols['colcat_model'],
         ,'coly':             []        # cols['coly']
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }   ### Filter data

         }}
    return model_dict



############################################################################
########## Train ###########################################################
def train():
    from source import run_train

    run_train.run_train(config_model_name = 'titanic_lightgbm',
                        path_data         =  'data/input/titanic/train/',
                        path_output       =  'data/output/titanic/a01_lightgbm/',
                        path_config_model =  root + "/titanic_classifier.py" , n_sample = -1)


###################################################################################
######### Check model #############################################################
def check():
    #### Load model
    from source.util_feature import load
    from source.models import model_sklearn as modelx
    import sys
    from source import models
    sys.modules['models'] = models
    model_tag =  "a01_lightgbm"


    dir_model    = dir_data + f"/output/titanic/{model_tag}/"
    modelx.model = load( dir_model + "/model/model.pkl" )
    stats        = load( dir_model + "/model/info.pkl" )
    colsX        = load( dir_model + "/model/colsX.pkl"   )
    coly         = load( dir_model + "/model/coly.pkl"   )
    print(stats)
    print(modelx.model.model)

    ### Metrics on test data
    stats['metrics_test']

    #### Loading training data  #######################################################
    dfX     = pd.read_csv(dir_model + "/check/dfX.csv")
    dfy     = dfX[coly]
    colused = colsX

    dfXtest = pd.read_csv(dir_model + "/check/dfXtest.csv")
    dfytest = dfXtest[coly]
    print(dfX.shape,  dfXtest.shape )


    #### Feature importance on training data
    lgb_featimpt_train,_ = util_feature.feature_importance_perm(modelx, dfX[colused], dfy, colused, n_repeats=1,
                                                                scoring='accuracy' )

    print(lgb_featimpt_train)

    #! python source/run_inference.py  run_predict  --config_model_name  LGBMRegressor  --n_sample 1000   --path_model /data/output/a01_lightgbm_huber/    --path_output /data/output/pred_a01_lightgbm_huber/    --path_data /data/input/train/



########################################################################################
####### Inference ######################################################################
def predict():
    from source import run_inference
    run_inference.run_predict('LGBMClassifier',
                              path_model  = '/data/output/titanic/a01_lightgbm/',
                              path_data   = '/data/input/titanic/test/',
                              path_output = '/data/output/titanic/pred_a01_titanic_lightgbm/',
                              n_sample    = -1)



if __name__ == "__main__":
    import fire
    fire.Fire()

