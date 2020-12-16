# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to titatinic dataet
All in one file config
!  python titanic_classifier.py  train
!  python titanic_classifier.py  check
!  python titanic_classifier.py  predict
"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')

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


def global_pars_update(model_dict,  data_name, config_name):
    global path_config_model, path_model, path_data_train, path_data_test, path_output_pred, n_sample,model_name
    model_name        = model_dict['model_pars']['config_model_name']
    path_config_model = root + f"/{config_file}"
    path_model        = f'data/output/{data_name}/a01_{model_name}/'
    path_data_train   = f'data/input/{data_name}/train/'
    path_data_test    = f'data/input/{data_name}/test/'
    path_output_pred  = f'/data/output/{data_name}/pred_a01_{config_name}/'
    n_sample          = model_dict['data_pars'].get('n_sample', 5000)

    model_dict[ 'global_pars'] = {}
    model_dict['global_pars']['config_name'] = config_name
    global_pars = [  'model_name', 'path_config_model', 'path_model', 'path_data_train',
                   'path_data_test', 'path_output_pred', 'n_sample'
                  ]
    for t in global_pars:
      model_dict['global_pars'][t] = globals()[t]
    return model_dict


def os_get_function_name():
    import sys
    return sys._getframe(1).f_code.co_name


####################################################################################
config_file     = "titanic_classifier.py"   ### name of file which contains data configuration
config_default  = 'titanic_lightgbm'   ### name of function which contains data configuration


config_name  = 'titanic_lightgbm'   ### name  of function which contains data configuration
n_sample     = 2000




####################################################################################
##### Params########################################################################
# data_name    = "titanic"     ### in data/input/

cols_input_type_1 = {
     "coly"   :   "Survived"
    ,"colid"  :   "PassengerId"
    ,"colcat" :   ["Sex", "Embarked" ]
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
       Contains all needed informations for Light GBM Classifier model,
       used for titanic classification task
    """
    data_name    = "titanic"     ### in data/input/
    model_name   = 'LGBMClassifier'
    n_sample     = 1000


    def post_process_fun(y):
        ### After prediction is done
        return  int(y)


    def pre_process_fun(y):
        ### Before the prediction is done
        return  int(y)


    model_dict = {'model_pars': {
        'model_path'       : path_model_out

        ### LightGBM API model   #######################################
        ,'config_model_name': model_name    ## ACTUAL Class name for model_sklearn.py
        ,'model_pars'       : {'objective': 'binary',
                               'n_estimators':3000,
                               'learning_rate':0.001,
                               'boosting_type':'gbdt',     ### Model hyperparameters
                                'early_stopping_rounds': 5
                              }

        ### After prediction  ##########################################
        , 'post_process_fun' : post_process_fun


        ### Before training  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,


                ### Pipeline for data processing ########################
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

      'data_pars': { 'n_sample' : n_sample,

          'cols_input_type' : cols_input_type_1,


          ### family of columns for MODEL  ########################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          ##  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate',
          #  'coltext',
          'cols_model_group': [ 'colnum_bin',
                               'colcat_bin'
                               ]


          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict





def titanic_lightgbm2(path_model_out="") :
    """
        python titanic_classifier.py preprocess --nsample 100  --config titanic_lightgbm2


    """
    data_name    = "titanic"     ### in data/input/
    model_name   = 'LGBMClassifier'
    n_sample     = 1000


    def post_process_fun(y):
        ### After prediction is done
        return  int(y)


    def pre_process_fun(y):
        ### Before the prediction is done
        return  int(y)


    model_dict = {'model_pars': {
        'model_path'       : path_model_out

        ### LightGBM API model   #######################################
        ,'config_model_name': model_name    ## ACTUAL Class name for model_sklearn.py
        ,'model_pars'       : {'objective': 'binary',
                               'n_estimators':10,
                               'learning_rate':0.01,
                               'boosting_type':'gbdt'  ### Model hyperparameters

                              }

        ### After prediction  ##########################################
        , 'post_process_fun' : post_process_fun


        ### Before prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,


    #### Default Pipeline Execution
    'pipe_list' : [
      {'uri' : 'source/preprocessors.py::pd_filter_rows',         'pars': {}, 'cols_family': 'colall',     'cols_out':'colall',        'type': 'filter' },
      {'uri' : 'source/preprocessors.py::pd_coly',                'pars': {}, 'cols_family': 'coly',       'cols_out':'coly',          'type': 'coly' },

      {'uri' : 'source/preprocessors.py::pd_colnum_bin',          'pars': {}, 'cols_family': 'colnum',     'cols_out':'colnum_bin',    'type': '' },
      {'uri' : 'source/preprocessors.py::pd_colnum_binto_onehot', 'pars': {}, 'cols_family': 'colnum_bin', 'cols_out':'colnum_onehot', 'type': '' },
      {'uri':  'source/preprocessors.py::pd_colcat_bin',          'pars': {}, 'cols_family': 'colcat',     'cols_out':'colcat_bin',    'type': ''},
      {'uri':  'source/preprocessors.py::pd_colcat_to_onehot',    'pars': {}, 'cols_family': 'colcat_bin', 'cols_out':'colcat_onehot', 'type': ''},

      {'uri' : 'source/preprocessors.py::pd_colcross',            'pars': {}, 'cols_family': 'colcross',   'cols_out':'colcross_hot',  'type': 'cross' }
    ],


               }
        },

      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      ,'early_stopping_rounds':5},

      'data_pars': {
          'cols_input_type' : cols_input_type_1,

          ### family of columns for MODEL  ########################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          ##  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate',
          #  'coltext',
          'cols_model_group': [ 'colnum', 'colcat_bin']


          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict





#####################################################################################
########## Profile data #############################################################
def data_profile(path_data_train="", path_model="", n_sample= 5000):
   from source.run_feature_profile import run_profile
   run_profile(path_data   = path_data_train,
               path_output = path_model + "/profile/",
               n_sample    = n_sample,
              )



###################################################################################
########## Preprocess #############################################################
def preprocess(config=None, nsample=None):
    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_preprocess2, run_preprocess
    run_preprocess2.run_preprocess(model_name     =  config_name,
                                path_data         =  m['path_data_train'],
                                path_output       =  m['path_model'],
                                path_config_model =  m['path_config_model'],
                                n_sample          =  nsample if nsample is not None else m['n_sample'],
                                mode              =  'run_preprocess')


##################################################################################
########## Train #################################################################
def train(config=None, nsample=None):

    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_train
    run_train.run_train(config_model_name =  config_name,
                        path_data         =  m['path_data_train'],
                        path_output       =  m['path_model'],
                        path_config_model =  m['path_config_model'] ,
                        n_sample          =  nsample if nsample is not None else m['n_sample'])


###################################################################################
######### Check data ##############################################################
def check():
   pass




####################################################################################
####### Inference ##################################################################
def predict(config=None, nsample=None):
    config_name  =  config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict['data_pars']['cols_input_type'])
    print(m)

    from source import run_inference,run_inference2
    run_inference2.run_predict(model_name,
                            path_model  = m['path_model'],
                            path_data   = m['path_data_test'],
                            path_output = m['path_output_pred'],
                            cols_group=mdict['data_pars']['cols_input_type'],
                            n_sample    = nsample if nsample is not None else m['n_sample'])

                            )


def run_all():
    data_profile()
    preprocess()
    train()
    check()
    predict()





###########################################################################################################
###########################################################################################################
"""
python  titanic_classifier.py  data_profile
python  titanic_classifier.py  preprocess  --nsample 100
python  titanic_classifier.py  train       --nsample 200
python  titanic_classifier.py  check
python  titanic_classifier.py  predict
python  titanic_classifier.py  run_all


"""
if __name__ == "__main__":

    import fire
    fire.Fire()
    


