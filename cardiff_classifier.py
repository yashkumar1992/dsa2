# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to cardif dataet
All in one file config

!  python cardiff_classifier.py  train
!  python cardiff_classifier.py  check
!  python cardiff_classifier.py  predict



"""
import warnings, copy
warnings.filterwarnings('ignore')
import os, sys
import pandas as pd

############################################################################
from source import util_feature
from source import run_preprocess


###### Path #########################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


####################################################################################
config_file  = "cardiff_classifier.py"
data_name    = "cardif"


config_name  = 'cardif_lightgbm'





####################################################################################
##### Params #######################################################################
def cardif_lightgbm(path_model_out="") :
    """
       cardif
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
        , 'model_pars'       : {'objective': 'binary','learning_rate':0.1,'boosting_type':'gbdt' }  # default
        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  None ,
                               }
                                 },
      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },

      'data_pars': {
          'cols_input_type' : {
                     "coly"   :   "target"
                    ,"colid"  :   "ID"
                    ,"colcat" :   ["v3","v30", "v31", "v47", "v52", "v56", "v66", "v71", "v74", "v75", "v79", "v91", "v107", "v110", "v112", "v113", "v125"]
                    ,"colnum" :   ["v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v23", "v25", "v26", "v27", "v28", "v29", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v48", "v49", "v50", "v51", "v53", "v54", "v55", "v57", "v58", "v59", "v60", "v61", "v62", "v63", "v64", "v65", "v67", "v68", "v69", "v70", "v72", "v73", "v76", "v77", "v78", "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89", "v90", "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v108", "v109", "v111", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v126", "v127", "v128", "v129", "v130", "v131"]
                    ,"coltext" :  []
                    ,"coldate" :  []
                    ,"colcross" : ["v3"]   #when 
                   },

          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']
         ,'cols_model':       []  # cols['colcat_model'],
         ,'coly':             []        # cols['coly']
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }   ### Filter data

         }}
    return model_dict




def cardif_sklearn(path_model_out="") :
   pass






####################################################################################################
########## Init variable ###########################################################################
globals()[config_name]()





###################################################################################
########## Train ##################################################################
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



###########################################################################################################
###########################################################################################################
"""
python  cardiff_classifier.py  train
python  cardiff_classifier.py  check
python  cardiff_classifier.py  predict
python  cardiff_classifier.py  run_all
"""
if __name__ == "__main__":
    import fire
    fire.Fire()


run_preprocess.run_preprocess(model_name='cardif_lightgbm', 
                              path_data=f'data/input/{data_name}/train/', 
                              path_output=f'data/output/{data_name}/a01_{model_name}/', 
                              path_config_model="source/config_model.py", 
                              n_sample=5000,
                              mode='run_preprocess'
                              )

