# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
	python  income_status_classifier.py  data_profile
	python  income_status_classifier.py  preprocess  --nsample 32560
	python  income_status_classifier.py  train       --nsample 32560
	python  income_status_classifier.py  check
	python  income_status_classifier.py  predict
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
    m                      = {}
    model_name             = model_dict['model_pars']['model_class']
    m['path_config_model'] = root + f"/{config_file}"
    m['config_name']       = config_name

    m['path_data_train']   = f'data/input/{data_name}/train/'
    m['path_data_test']    = f'data/input/{data_name}/test/'

    m['path_model']        = f'data/output/{data_name}/{config_name}/'
    m['path_output_pred']  = f'data/output/{data_name}/pred_{config_name}/'
    m['n_sample']          = model_dict['data_pars'].get('n_sample', 32560)

    model_dict[ 'global_pars'] = m
    return model_dict
    


def os_get_function_name():
    import sys
    return sys._getframe(1).f_code.co_name


####################################################################################
config_file     = "income_status_classifier.py"   ### name of file which contains data configuration
config_default  = 'income_status_lightgbm'        ### name of function which contains data configuration




####################################################################################
##### Params########################################################################
# data_name    = "income_status"     ### in data/input/
cols_input_type_1 = {
     "coly"   :   "status"
    ,"colid"  :   "id"
    ,"colcat" :   ["occupation","workclass","native-country","education-num","marital-status","relationship","race","sex"]
    ,"colnum" :   ["age", "final_weight", "capital-gain", "capital-loss", "hours-per-week"]
    ,"coltext" :  []
    ,"coldate" :  []
    ,"colcross" : ["occupation","workclass","native-country","education-num","marital-status","relationship","race","sex","age", "final_weight", "capital-gain", "capital-loss", "hours-per-week"]
}


cols_input_type_2 = {
     "coly"   :   "status"
    ,"colid"  :   "id"
    ,"colcat" :   ["occupation","workclass","native-country","education-num","marital-status","relationship","race","sex"]
    ,"colnum" :   ["age", "final_weight", "capital-gain", "capital-loss", "hours-per-week"]
    ,"coltext" :  []
    ,"coldate" :  []
    ,"colcross" : ["occupation","workclass","native-country","education-num","marital-status","relationship","race","sex","age", "final_weight", "capital-gain", "capital-loss", "hours-per-week"]
}



####################################################################################
def income_status_lightgbm(path_model_out="") :
    """
       Contains all needed informations for Light GBM Classifier model,
       used for titanic classification task
    """
    data_name    = "income_status"         ### in data/input/
    model_class  = 'LGBMClassifier'  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 32560

    def post_process_fun(y):
        ### After prediction is done
        return  int(y)

    def pre_process_fun(y):
        ### Before the prediction is done
        return  int(y)


    model_dict = {'model_pars': {
        'model_path'       : path_model_out

        ### LightGBM API model   #######################################
        ,'model_class': model_class
        ,'model_pars' : {'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':1.0,
						'importance_type':'split', 'learning_rate':0.001, 'max_depth':-1,
						'min_child_samples':20, 'min_child_weight':0.001, 'min_split_gain':0,
						'n_estimators':5000, 'n_jobs':-1, 'num_leaves':31, 'objective':None,
						'random_state':None, 'reg_alpha':0, 'reg_lambda':0.0, 'silent':True,
						'subsample':1.0, 'subsample_for_bin':200000, 'subsample_freq':0}
                      
                        
        

        ### After prediction  ##########################################
        , 'post_process_fun' : post_process_fun


        ### Before training  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,


        ### Pipeline for data processing ##############################
        'pipe_list': [

            #{'uri': 'data/input/income/manual_preprocessing.py::pd_income_processor',      'pars': {}, 'cols_family': 'colall',   'cols_out': 'colall',
            #        'type': 'filter'         },


            {'uri': 'source/preprocessors.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/preprocessors.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
               }
        },

      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },

      'data_pars': { 'n_sample' : n_sample,
          'cols_input_type' : cols_input_type_1,

          ### family of columns for MODEL  ########################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate',
          #  'coltext',
          'cols_model_group': [ 'colnum_bin',
                                'colcat_bin',
                                # 'coltext',
                                # 'coldate',
                                # 'colcross_pair'
                              ]

          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    ############################################################
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

    from source import run_preprocess
    run_preprocess.run_preprocess(config_name       =  config_name,
                                  config_path       =  m['config_path'],
                                  n_sample          =  nsample if nsample is not None else m['n_sample'],

                                  ### Optonal
                                  mode              =  'run_preprocess')


##################################################################################
########## Train #################################################################
def train(config=None, nsample=None):

    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_train
    run_train.run_train(config_name       =  config_name,
                        config_path       =  m['config_path'],
                        n_sample          =  nsample if nsample is not None else m['n_sample'],
                        )


###################################################################################
######### Check data ##############################################################
def check():
   pass




####################################################################################
####### Inference ##################################################################
def predict(config=None, nsample=None):
    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']


    from source import run_inference
    run_inference.run_predict(config_name = config_name,
                              config_path =  m['config_path'],
                              n_sample    = nsample if nsample is not None else m['n_sample'],

                              #### Optional
                              path_data   = m['path_pred_data'],
                              path_output = m['path_pred_output'],
                              model_dict  = None
                              )





###########################################################################################################
###########################################################################################################
"""
python  income_status_classifier.py  data_profile
python  income_status_classifier.py  preprocess
python  income_status_classifier.py  train 
python  income_status_classifier.py  check
python  income_status_classifier.py  predict
python  income_status_classifier.py  run_all
"""
if __name__ == "__main__":
    import fire
    fire.Fire()