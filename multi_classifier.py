# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/tapioca/multiclass-lightgbm

https://medium.com/@nitin9809/lightgbm-binary-classification-multi-class-classification-regression-using-python-4f22032b36a2



All in one file config
!  python multiclass_classifier.py  train
!  python multiclass_classifier.py  check
!  python multiclass_classifier.py  predict
"""
import warnings, copy
warnings.filterwarnings('ignore')
import os, sys
import pandas as pd
import numpy as np
###################################################################################
from source import util_feature


###### Path ########################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


####################################################################################
config_file  = f"multi_classifier.py"
data_name    = f"multiclass"     ### in data/input/



config_name  = 'multiclass_lightgbm'
n_sample     =  6000


colid   = 'pet_id'
coly    = 'pet_category'
coldate = ['issue_date','listing_date']
colcat  = ['color_type']
colnum  = ['length(m)','height(cm)','condition','X1','X2','breed_category']
colcross= ['pet_id', 'issue_date', 'listing_date', 'condition', 'color_type','length(m)', 'height(cm)', 'X1', 'X2', 'breed_category']


cols_input_type_1 = {
                     "coly"   :   coly
                    ,"colid"  :   colid
                    ,"colcat" :   colcat
                    ,"colnum" :   colnum
                    ,"coltext" :  []
                    ,"coldate" :  []
                    ,"colcross" : colcross
                   }



####################################################################################
##### Params########################################################################
def multiclass_lightgbm(path_model_out="") :
    """
       multiclass
    """
    global path_config_model, path_model, path_data_train, path_data_test, path_output_pred, n_sample,model_name

    config_name       = 'multiclass_lightgbm'
    model_name        = 'LGBMClassifier'
    n_sample          = 6000

    def post_process_fun(y):
        ### After prediction is done
        return  y.astype('int')

    def pre_process_fun_multi(y):
        ### Before the prediction is done
        map_dict_={0:0,1:1,2:2,4:3}
        return  map_dict_[y]


    model_dict = {'model_pars': {
        'model_path'       : path_model_out

        ### LightGBM API model       ###################################
        ,'config_model_name': model_name    ## ACTUAL Class name for model_sklearn.py
        ,'model_pars'       : {'objective': 'multiclass','num_class':4,'metric':'multi_logloss',
                                'learning_rate':0.03,'boosting_type':'gbdt'


                               }

        ### After prediction  ##########################################
        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun_multi ,
                                
                                ### Pipeline for data processing.
                               'pipe_list'  : [ 'filter',     ### Fitler the data
                                                'label',      ### Normalize the label
                                                'dfnum_bin',
                                                'dfnum_hot',
                                                'dfcat_bin',
                                                'dfcat_hot',
                                                'dfcross_hot', ]
                               }
        },

      'compute_pars': { 'metric_list': ['roc_auc_score','accuracy_score'],
                        'probability': True,
                      },

      'data_pars': {
          'cols_input_type' : cols_input_type_1,

          ### used for the model input  ###############################
          # "colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']


          ### Filter data rows   #####################################
         ,'filter_pars': { 'ymax' : 5 ,'ymin' : -1 }

         }
      }

    ################################################################################################
    ##### Filling Global parameters    #############################################################
    path_config_model = root + f"/{config_file}"
    path_model        = f'data/output/{data_name}/a01_{model_name}/'
    path_data_train   = f'data/input/{data_name}/train/'
    path_data_test    = f'data/input/{data_name}/test/'
    path_output_pred  = f'/data/output/{data_name}/pred_a01_{config_name}/'

    model_dict[ 'global_pars'] = {}
    global_pars = [ 'config_name', 'model_name', 'path_config_model', 'path_model', 'path_data_train',
                   'path_data_test', 'path_output_pred', 'n_sample'
            ]
    for t in global_pars:
      model_dict['global_pars'][t] = globals()[t] 


    return model_dict







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
    from source import run_train
    run_train.run_data_check(path_output =  path_model,
                             scoring     =  'accuracy')



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
python  multiclass_classifier.py  data_profile
python  multiclass_classifier.py  preprocess
python  multiclass_classifier.py  train
python  multiclass_classifier.py  check
python  multiclass_classifier.py  predict
python  multiclass_classifier.py  run_all
"""
if __name__ == "__main__":
    import fire
    fire.Fire()
    
