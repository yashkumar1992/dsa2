# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Automatic Feature generator

"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')

####################################################################################
###### Path ########################################################################
root_repo      =  os.path.abspath(os.getcwd()).replace("\\", "/") + "/"     ; print(root_repo)
THIS_FILEPATH  =  os.path.abspath(__file__) 


sys.path.append(root_repo)
from source.util_feature import save,os_get_function_name


def global_pars_update(model_dict,  data_name, config_name):
    print("config_name", config_name)
    dir_data  = root_repo + "/data/"  ; print("dir_data", dir_data)

    m                      = {}
    m['config_path']       = THIS_FILEPATH  
    m['config_name']       = config_name

    #### peoprocess input path
    m['path_data_preprocess'] = dir_data + f'/input/{data_name}/train/'

    #### train input path
    m['path_data_train']      = dir_data + f'/input/{data_name}/train/'
    m['path_data_test']       = dir_data + f'/input/{data_name}/test/'
    #m['path_data_val']       = dir_data + f'/input/{data_name}/test/'

    #### train output path
    m['path_train_output']    = dir_data + f'/output/{data_name}/{config_name}/'
    m['path_train_model']     = dir_data + f'/output/{data_name}/{config_name}/model/'
    m['path_features_store']  = dir_data + f'/output/{data_name}/{config_name}/features_store/'
    m['path_pipeline']        = dir_data + f'/output/{data_name}/{config_name}/pipeline/'


    #### predict  input path
    m['path_pred_data']       = dir_data + f'/input/{data_name}/test/'
    m['path_pred_pipeline']   = dir_data + f'/output/{data_name}/{config_name}/pipeline/'
    m['path_pred_model']      = dir_data + f'/output/{data_name}/{config_name}/model/'

    #### predict  output path
    m['path_pred_output']     = dir_data + f'/output/{data_name}/pred_{config_name}/'

    #####  Generic
    m['n_sample']             = model_dict['data_pars'].get('n_sample', 5000)

    model_dict[ 'global_pars'] = m
    return model_dict


####################################################################################
##### Params########################################################################
config_default   = 'titanic1'          ### name of function which contains data configuration


cols_input_type_2 = {
     "coly"   :   "Survived"
    ,"colid"  :   "PassengerId"
    ,"colcat" :   ["Sex", "Embarked" ]
    ,"colnum" :   ["Pclass", "Age","SibSp", "Parch","Fare"]
    ,"coltext" :  ["Name", "Ticket"]
    ,"coldate" :  []
    ,"colcross" : [ "Name", "Sex", "Ticket","Embarked","Pclass", "Age","SibSp", "Parch","Fare" ]

    ,'colgen'  :  [   "Pclass", "Age","SibSp", "Parch","Fare" ]
}


#################################################################################


def titanic1(path_model_out="") :
    """
       Contains all needed informations for Light GBM Classifier model,
       used for titanic classification task
    """
    config_name  = os_get_function_name()
    data_name    = "titanic"         ### in data/input/
    model_class  = 'LGBMClassifier'  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 500

    def post_process_fun(y):
     return  int(y)

    def pre_process_fun(y):
     return  int(y)


    model_dict = {'model_pars': {
    ### LightGBM API model   ######################################
     'model_class': model_class
    ,'model_pars' : {'objective': 'binary', 'n_estimators':100,
                    }

    , 'post_process_fun' : post_process_fun
    , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,


    ### Pipeline for data processing ##############################
    'pipe_list': [
        {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
        {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },

        # {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
        {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },

        # {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
        # {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'},
    ],
    }
    },

  'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                  },

  'data_pars': { 'n_sample' : n_sample,
      'cols_input_type' : cols_input_type_2,
      ### family of columns for MODEL  ########################################################
      #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
      #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
      #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
      #  'coldate',
      #  'coltext',
      'cols_model_group': [ 'colnum',  ### should be optional 'colcat'
                            'colcat_bin',
                          ]

      ### Filter data rows   ##################################################################
     ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    ########################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name )
    return model_dict


def hyperparam_optim():
    from source.run_train import  run_train
    from source.run_hyperopt import run_hyper_optuna

    ####### Initial dict
    mdict = titanic1()
    
    def objective_fun(mdict):
        metric_name = "accuracy_score"
        print(mdict)
        ddict       = run_train(config_name="", config_path="", n_sample=5000,
                             mode="run_preprocess", model_dict=mdict,
                             return_mode='dict')
        print(ddict['stats']['metrics_test'].to_dict('records')[0])
        ddict['stats']['accuracy_score'] = ddict['stats']['metrics_test'].to_dict('records')[0]['metric_val']
        
        print(ddict)
        res = ddict['stats'][metric_name]
        return res
  

    ###### Range in parameters
    mdict_range =   {'model_pars': {
        ### LightGBM API model   #######################################
        'model_pars' : { 'objective': 'binary',
                         'n_estimators': ('int', 20, 100),
                       },
          }
    }
    
 

    

    engine_pars = {'metric_target':'loss'}
    result_p    = run_hyper_optuna(objective_fun, mdict, mdict_range, engine_pars, ntrials= 3)
    return result_p


###################################################################################
########## Preprocess #############################################################
### def preprocess(config='', nsample=1000):
from core_run import preprocess



##################################################################################
########## Train #################################################################
from core_run import train



####################################################################################
####### Inference ##################################################################
# predict(config='', nsample=10000)
from core_run import predict



###########################################################################################################
###########################################################################################################
"""
python  hyperopt.py  train   --nsample 200

python  hyperopt.py  hyperparam_optim


"""

def test():
    hyperparam_optim()
    



if __name__ == "__main__":
    import fire
    fire.Fire()
    #test()


