# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
cd analysis

python source/run_train.py  run_train --config_model_name elasticnet  --path_data data/input/train/    --path_output data/output/a01_elasticnet/

! activate py36 && python source/run_train.py  run_train   --n_sample 100  --config_model_name lightgbm  --path_model_config source/config_model.py  --path_output /data/output/a01_test/     --path_data /data/input/train/








"""
import warnings
warnings.filterwarnings('ignore')
import sys
import gc
import os
import logging
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import json
import pickle
import scipy
import importlib

# from tqdm import tqdm_notebook
import cloudpickle as pickle
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve



#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")
import util_feature


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


# from diskcache import Cache
# cache = Cache('db.cache')
# cache.reset('size_limit', int(2e9))


####################################################################################################
####################################################################################################
def log(*s, n=0, m=0):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)

from util_feature import  load_function_uri, save, load, save_list


####################################################################################################
####################################################################################################
def load_dataset(path_train_X, path_train_y, colid, n_sample=-1):
    df = pd.read_csv(path_train_X)
    if n_sample > 0:
        df = df.sample(frac=1.0)
        df = df.iloc[:n_sample, :]
    try :
      dfy = pd.read_csv(path_train_y)
      df = df.join(dfy.set_index(colid), on=colid, how="left")
    except : 
      pass  
    df = df.set_index(colid)
    return df


def save_features(df, name, path):
    if path is not None :
       os.makedirs( f"{path}/{name}", exist_ok=True)

       df.to_parquet( f"{path}/{name}/features.parquet")


from run_preprocess import  preprocess


####################################################################################################
##### train    #####################################################################################
def map_model(model_name):
    try :
       ##  'models.model_bayesian_pyro'   'model_widedeep'
       mod    = f'models.{model_name}'
       modelx = importlib.import_module(mod) 
       
    except :
        ### Al SKLEARN API
        #['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
       mod    = 'models.model_sklearn'
       modelx = importlib.import_module(mod) 
    
    return modelx


def train(model_dict, dfX, cols_family, post_process_fun):
    """
    """
    model_pars, compute_pars = model_dict['model_pars'], model_dict['compute_pars']
    data_pars = model_dict['data_pars']
    model_name, model_path = model_pars['config_model_name'], model_pars['model_path']
    metric_list = compute_pars['metric_list']

    log("#### Data preparation #############################################################")
    log(dfX.shape)
    dfX = dfX.sample(frac=1.0)
    itrain = int(0.6 * len(dfX))
    ival   = int(0.8 * len(dfX))
    colid = cols_family['colid']
    colsX = data_pars['cols_model']
    coly  = data_pars['coly']
    data_pars['data_type'] = 'ram'
    data_pars['train'] = {'Xtrain': dfX[colsX].iloc[:itrain, :],
                          'ytrain': dfX[coly].iloc[:itrain],
                          'Xtest':  dfX[colsX].iloc[itrain:ival, :],
                          'ytest':  dfX[coly].iloc[itrain:ival],

                          'Xval':   dfX[colsX].iloc[ival:, :],
                          'yval':   dfX[coly].iloc[ival:],
                          }
    
    log("#### Model Instance ##########################################################")
    # from config_model import map_model    
    modelx = map_model(model_name)    
    log(modelx)
    modelx.reset()
    modelx.init(model_pars, compute_pars=compute_pars)
    modelx.fit(data_pars, compute_pars)

    log("#### Metrics #################################################################")
    from util_feature import  sk_metrics_eval

    stats = {}
    ypred               = modelx.predict(dfX[colsX], compute_pars=compute_pars)
    dfX[coly + '_pred'] = ypred  # y_norm(ypred, inverse=True)
    dfX[coly]           = post_process_fun(dfX[coly].values)
    # dfX[coly] = dfX[coly].values.astype('int64')

    metrics_test = sk_metrics_eval(metric_list,
                                   ytrue= dfX[coly].iloc[ival:],
                                   ypred= dfX[coly + '_pred'].iloc[ival:], )
    stats['metrics_test'] = metrics_test
    log(stats)

    log("saving model, dfX, columns", model_path)
    os.makedirs(model_path, exist_ok=True)
    modelx.save(model_path, stats)
    
    log(modelx.model.model_pars, modelx.model.compute_pars)
    a = load(model_path + "/model.pkl")
    log("check re-loaded", a.model_pars)
    
    save_list(model_path, ['colsX', 'coly'], locals())
    return dfX.iloc[:ival, :].reset_index(), dfX.iloc[ival:, :].reset_index()


####################################################################################################
############CLI Command ############################################################################
def run_train(config_model_name, path_data, path_output, path_config_model="source/config_model.py", n_sample=5000,
              run_preprocess=1, use_feature_store= False):
    """
      Configuration of the model is in config_model.py file

    """
    path_output       = root + path_output
    path_data         = root + path_data
    path_pipeline_out = path_output + "/pipeline/"
    path_model_out    = path_output + "/model/"
    path_check_out    = path_output + "/check/"
    path_train_X      = path_data   + "/features.zip"
    path_train_y      = path_data   + "/target.zip"
    path_train_features = path_output + '/features_store/'
    log(path_output)

    log("#### Model Dynamic loading  ######################################################")
    model_dict_fun = load_function_uri(uri_name=path_config_model + "::" + config_model_name)
    # model_dict_fun = getattr(importlib.import_module("config_model"), config_model_name)
    model_dict     = model_dict_fun(path_model_out)   ### params


    log("#### load input column family  ###################################################")
    try :
        cols_group = model_dict['data_pars']['cols_input_type']  ### the model config file
    except :
        cols_group = json.load(open(path_data + "/cols_group.json", mode='r'))
    log(cols_group)


    log("#### Preprocess  #################################################################")
    preprocess_pars = model_dict['model_pars']['pre_process_pars']
    filter_pars     = model_dict['data_pars']['filter_pars']

    if run_preprocess :
        dfXy, cols      = preprocess(path_train_X, path_train_y, path_pipeline_out, cols_group, n_sample,
                                 preprocess_pars, filter_pars, path_train_features)
    model_dict['data_pars']['coly'] = cols['coly']

    if use_feature_store:
        dfXy=pd.read_parquet(path_train_features + "/dfX/features.parquet")
    
    ### Get actual column names from colum groups : colnum , colcat
    model_dict['data_pars']['cols_model'] = sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , [])                
    log(  model_dict['data_pars']['cols_model'] , model_dict['data_pars']['coly'])
    
   
    log("######### Train model: ###########################################################")
    log(str(model_dict)[:1000])
    post_process_fun = model_dict['model_pars']['post_process_fun']    
    dfXy, dfXytest   = train(model_dict, dfXy, cols, post_process_fun)


    log("######### export #################################", )
    os.makedirs(path_check_out, exist_ok=True)
    colexport = [cols['colid'], cols['coly'], cols['coly'] + "_pred"]
    dfXy[colexport].to_csv(path_check_out + "/pred_check.csv")  # Only results
    dfXy.to_parquet(path_check_out + "/dfX.parquet")  # train input data generate parquet
    #dfXy.to_csv(path_check_out + "/dfX.csv")  # train input data generate csv
    dfXytest.to_parquet(path_check_out + "/dfXtest.parquet")  # Test input data  generate parquet
    #dfXytest.to_csv(path_check_out + "/dfXtest.csv")  # Test input data  generate csv
    log("######### finish #################################", )


if __name__ == "__main__":
    import fire
    fire.Fire()




