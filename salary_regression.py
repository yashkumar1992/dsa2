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


import copy

##############################################################################################   
def y_norm(y, inverse=True, mode='boxcox'):
    ## Normalize the input/output
    if mode == 'boxcox':
        width0 = 53.0  # 0,1 factor
        k1 = 0.6145279599674994  # Optimal boxCox lambda for y
        if inverse:
            y2 = y * width0
            y2 = ((y2 * k1) + 1) ** (1 / k1)
            return y2
        else:
            y1 = (y ** k1 - 1) / k1
            y1 = y1 / width0
            return y1

    if mode == 'norm':
        m0, width0 = 0.0, 350.0  ## Min, Max
        if inverse:
            y1 = (y * width0 + m0)
            return y1

        else:
            y2 = (y - m0) / width0
            return y2
    else:
        return y



##############################################################################################  
def salary_elasticnetcv(path_model_out):
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'config_model_name': 'ElasticNetCV'
        , 'model_path': path_model_out
        , 'model_pars': {}  # default ones
        , 'post_process_fun': post_process_fun 
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun,
                               }
                                 },
      'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                       'explained_variance_score', 'r2_score', 'median_absolute_error']
                      },
      'data_pars': {
          'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]    
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                  }}
    return model_dict               


def salary_lightgbm(path_model_out) :
    """
      Huber Loss includes L1  regurarlization         
      We test different features combinaison, default params is optimal
    """
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'config_model_name': 'LGBMRegressor'
        , 'model_path': path_model_out
        , 'model_pars': {'objective': 'huber', }  # default
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,
                               }
                                 },
      'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                        'explained_variance_score', 'r2_score', 'median_absolute_error']
                      },
    
      'data_pars': {
          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']    
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
         
         }}
    return model_dict               



def salary_lightgbm(path_model_out) :
    """
      Huber Loss includes L1  regurarlization         
      We test different features combinaison, default params is optimal
    """
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'config_model_name': 'LGBMRegressor'
        , 'model_path': path_model_out
        , 'model_pars': {'objective': 'huber', }  # default
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,
                               }
                                 },
      'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                        'explained_variance_score', 'r2_score', 'median_absolute_error']
                      },
    
      'data_pars': {
          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']    
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
         
         }}
    return model_dict    

def salary_bayesian_pyro(path_model_out) :
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'config_model_name': 'model_bayesian_pyro'
        , 'model_path': path_model_out
        , 'model_pars': {'input_width': 112, }  # default
        , 'post_process_fun': post_process_fun
                                 
        , 'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,
                               }
                                 },          
                  
      'compute_pars': {'compute_pars': {'n_iter': 1200, 'learning_rate': 0.01}
                     , 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                       'explained_variance_score', 'r2_score', 'median_absolute_error']
                     , 'max_size': 1000000
                     , 'num_samples': 300
       },
      'data_pars': {
          'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]    
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                  }}
    return model_dict               
                

def salary_glm( path_model_out) :
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')



    model_dict = {'model_pars': {'config_model_name': 'TweedieRegressor'  # Ridge
        , 'model_path': path_model_out
        , 'model_pars': {'power': 0, 'link': 'identity'}  # default ones
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun }
                                 },
                  'compute_pars': {'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                                   'explained_variance_score',  'r2_score', 'median_absolute_error']
                                  },
      'data_pars': {
          'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                  }
    }
    return model_dict               
               




###### Path ################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)



############################################################################
########## Train ###########################################################
def train():
    from source import run_train

    run_train.run_train(config_model_name =  'titanic_lightgbm',
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
