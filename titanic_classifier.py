# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to titatinic dataet
All in one file config
!  python titanic_classifier.py  train
!  python titanic_classifier.py  check
!  python titanic_classifier.py  predict
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


####################################################################################
config_file  = "titanic_classifier.py"
data_name    = "titanic"


config_name  = 'titanic_lightgbm'





####################################################################################
##### Params########################################################################
def titanic_lightgbm(path_model_out="") :
    """
       titanic
    """
    global path_config_model, path_model, path_data_train, path_data_test, path_output_pred, n_sample,model_name

    config_name       = 'cardif_lightgbm'
    model_name        = 'LGBMClassifier'

    path_config_model = root + f"/{config_file}"
    path_model        = f'data/output/{data_name}/a01_{model_name}/'
    path_data_train   = f'data/input/{data_name}/train/'
    path_data_test    = f'data/input/{data_name}/test/'
    path_output_pred  = f'/data/output/{data_name}/pred_a01_cardif_lightgbm/'
    n_sample          = -1
    

    def post_process_fun(y):
        ### After prediction is done
        return  y.astype('int')

    def pre_process_fun(y):
        ### Before the prediction is done
        return  y.astype('int')

    model_dict = {'model_pars': {'config_model_name': 'LGBMClassifier'    ## Class name for model_sklearn.py
        , 'model_path'       : path_model_out
        , 'model_pars'       : {'objective': 'binary','learning_rate':0.03,'boosting_type':'gbdt' }  # default
        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  None ,
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




def titanic_sklearn(path_model_out="") :
   pass






####################################################################################################
########## Init variable ###########################################################################
globals()[config_name]()

###################################################################################
########## Preprocess #############################################################
def preprocess():
    from source import run_preprocess

    run_preprocess.run_preprocess(model_name        =  config_name, 
                                  path_data         =  path_data_train, 
                                  path_output       =  path_model, 
                                  path_config_model =  path_config_model, 
                                  n_sample          =  n_sample,
                                  mode              =  'run_preprocess')

############################################################################
########## Train ###########################################################
def train():
    from source import run_train

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
        ddfX     = pd.read_csv(dir_model + "/check/dfX.csv")  #to load csv
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
    preprocess()
    train()
    check()
    predict()



###########################################################################################################
###########################################################################################################
"""
python  titanic_classifier.py  preprocess
python  titanic_classifier.py  train
python  titanic_classifier.py  check
python  titanic_classifier.py  predict
python  titanic_classifier.py  run_all
"""
if __name__ == "__main__":
    import fire
    fire.Fire()
    
"""


import template_run
template_run.config_name       = config_name
template_run.path_config_model = path_config_model
template_run.path_model        = path_model
template_run.path_data_train   = path_data_train
template_run.path_data_test    = path_data_test
template_run.path_output_pred  = path_output_pred
template_run.n_sample          = n_sample
template_run.model_name        = model_name

print( template_run.config_name )
train                          = template_run.train
predict                        = template_run.predict
run_all                        = template_run.run_all



"""
    
    
    
