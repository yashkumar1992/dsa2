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

###################################################################################
from source import util_feature

####################################################################################
###### Path ########################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)



####################################################################################
config_file  = "titanic_classifier.py"   ### name of file which contains data configuration
data_name    = "titanic"     ### in data/input/



config_name  = 'titanic_lightgbm'   ### name of function which contains data configuration 
n_sample     = 2000




####################################################################################
##### Params########################################################################
cols_input_type_1 = {
     "coly"   :   "Survived"
    ,"colid"  :   "PassengerId"
    ,"colcat" :   [  "Sex", "Embarked" ]
    ,"colnum" :   ["Pclass", "Age","SibSp", "Parch","Fare"]
    ,"coltext" :  ["Name", "Ticket"]
    ,"coldate" :  []
    ,"colcross" : [ "Name", "Sex", "Ticket","Embarked","Pclass", "Age","SibSp", "Parch","Fare" ]
}


### family of columns for MODEL  ########################################################
""" 
    'colid',
    "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns                        
    "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns                        
    'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns            
    'coldate',
    'coltext',            
    "coly"
"""



####################################################################################
def titanic_lightgbm(path_model_out="") :
    """
       Contains all needed informations for Light GBM Classifier model, used for titanic classification task
    """
    global path_config_model, path_model, path_data_train, path_data_test, path_output_pred, n_sample,model_name

    config_name       = 'titanic_lightgbm'
    model_name        = 'LGBMClassifier'

    path_config_model = root + f"/{config_file}"
    path_model        = f'data/output/{data_name}/a01_{model_name}/'
    path_data_train   = f'data/input/{data_name}/train/'
    path_data_test    = f'data/input/{data_name}/test/'
    path_output_pred  = f'/data/output/{data_name}/pred_a01_{config_name}/'

    n_sample    = 1000


    def post_process_fun(y):
        ### After prediction is done
        return  y.astype('int')

    def pre_process_fun(y):
        ### Before the prediction is done
        return  y.astype('int')

    model_dict = {'model_pars': {
        'model_path'       : path_model_out

        ### LightGBM API model       ###################################
        ,'config_model_name': model_name    ## ACTUAL Class name for model_sklearn.py
        ,'model_pars'       : {'objective': 'binary',
                                'learning_rate':0.03,'boosting_type':'gbdt'    ### Model hyperparameters

                              }

        ### After prediction  ##########################################
        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  None ,
                                
                ### Pipeline for data processing.
               'pipe_list'  : [ 'filter',     ### Fitler the data
                                'label',      ### Normalize the label
                                'dfnum_bin',  ### Create bins for numerical columns
                                'dfnum_hot',  ### One hot encoding for numerical columns
                                'dfcat_bin',  ### Create bins for categorical columns
                                'dfcat_hot',  ### One hot encoding for categorical columns
                                'dfcross_hot', ]   ### Crossing of features which are one hot encoded
               }
        },
      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },

      'data_pars': {
          'cols_input_type' : cols_input_type_1,


          ### family of columns for MODEL  ########################################################
          'cols_model_group': [ 'colnum', 'colcat_bin']


          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }

     ,'global_pars' : {}
      }

    ################################################################################################
    ##### Filling Global parameters    #############################################################
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
    """
    Preprocessing of input data, in order to prepare them for training

    """
    import run
    run.preprocess(config_uri = config_file + '::' + config_name)


########## Train ###########################################################
def train():

    """
    Splits preprocessed data into train and test, and fits them in the model

    """
    import run
    run.train(config_uri = config_file + '::' + config_name)

######### Check model #############################################################
def check():
    """
    It runs trained model and gives feature imporance graph as ouput

    """
    import run
    from source import run_train
    run.train(config_uri = config_file + '::' + config_name)

    #! python source/run_inference.py  run_predict  --config_model_name  LGBMRegressor  --n_sample 1000   --path_model /data/output/a01_lightgbm_huber/    --path_output /data/output/pred_a01_lightgbm_huber/    --path_data /data/input/train/



####### Inference ######################################################################
def predict():
    """
    Creates csv file with predictions

    """
    import run
    run.predict(config_uri = config_file + '::' + config_name)



def run_all():
    """

    Function which runs all previous functions, in order to perform all tasks, from preprocessing of data to final prediction

    """
    preprocess()
    train()
    check()
    predict()



###########################################################################################################
###########################################################################################################
"""
python  titanic_classifier.py  data_profile
python  titanic_classifier.py  preprocess
python  titanic_classifier.py  train
python  titanic_classifier.py  check
python  titanic_classifier.py  predict
python  titanic_classifier.py  run_all
"""
if __name__ == "__main__":
    import fire
    fire.Fire()
    
