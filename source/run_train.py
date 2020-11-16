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


def save_list(path, name_list, glob):
    import pickle, os
    os.makedirs(path, exist_ok=True)
    for t in name_list:
        log(t, f'{path}/{t}.pkl')
        pickle.dump(glob[t], open(f'{path}/{t}.pkl', mode='wb'))


def save(obj, path):
    import cloudpickle as pickle, os
    if os.path.isfile(path) :
       os.makedirs(  os.path.dirname( path), exist_ok=True)
    log(f'{path}')
    pickle.dump(obj, open(f'{path}', mode='wb'))


def load(file_name):
    import cloudpickle  as pickle
    return pickle.load(open(f'{file_name}', mode='rb'))


def load_function_uri(uri_name="path_norm"):
    """
    #load dynamically function from URI
    ###### Pandas CSV case : Custom MLMODELS One
    #"dataset"        : "mlmodels.preprocess.generic:pandasDataset"
    ###### External File processor :
    #"dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
    """
    
    import importlib, sys
    from pathlib import Path
    pkg = uri_name.split("::")

    assert len(pkg) > 1, "  Missing :   in  uri_name module_name:function_or_class "
    package, name = pkg[0], pkg[1]
    
    try:
        #### Import from package mlmodels sub-folder
        return  getattr(importlib.import_module(package), name)

    except Exception as e1:
        try:
            ### Add Folder to Path and Load absoluate path module
            path_parent = str(Path(package).parent.parent.absolute())
            sys.path.append(path_parent)
            log(path_parent)

            #### import Absolute Path model_tf.1_lstm
            model_name   = Path(package).stem  # remove .py
            package_name = str(Path(package).parts[-2]) + "." + str(model_name)
            #log(package_name, config_model_name)
            return  getattr(importlib.import_module(package_name), name)

        except Exception as e2:
            raise NameError(f"Module {pkg} notfound, {e1}, {e2}")


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
       os.makedirs( f"{path}/{name}" )
       df.to_parquet( f"{path}/{name}/features.parquet")


# @cache.memoize(typed=True,  tag='fib')  ### allow caching results
def preprocess(path_train_X="", path_train_y="", path_pipeline_export="", cols_group=None, n_sample=5000,
               preprocess_pars={}, filter_pars={}, path_train_features=None):
    """
      FUNCTIONNAL approach is used for pre-processing pipeline, (vs sklearn tranformer class..)
      so the code can be EASILY extensible to PYSPPARK.
      PYSPARK  supports better UDF, lambda function.
      Pyspark cannot support Class type processing on Dataframe (ie sklearn transformer class)
      
    """
    from util_feature import (pd_colnum_tocat, pd_col_to_onehot, pd_colcat_mapping, pd_colcat_toint,
                              pd_feature_generate_cross)

    ##### column names for feature generation ###############################################
    log(cols_group)
    coly            = cols_group['coly']  # 'salary'
    colid           = cols_group['colid']  # "jobId"
    colcat          = cols_group['colcat']  # [ 'companyId', 'jobType', 'degree', 'major', 'industry' ]
    colnum          = cols_group['colnum']  # ['yearsExperience', 'milesFromMetropolis']
    
    colcross_single = cols_group.get('colcross', [])   ### List of single columns
    #coltext        = cols_group.get('coltext', [])
    coltext         = cols_group['coltext']
    coldate         = cols_group.get('coldate', [])
    colall          = colnum + colcat + coltext + coldate
    log(colall)

    ##### Load data ########################################################################
    df =load_dataset(path_train_X, path_train_y, colid, n_sample= n_sample)


    ##### Filtering / cleaning rows :   ####################################################
    ymin, ymax = filter_pars.get('ymin', -9999999999.0), filter_pars.get('ymax', 999999999.0) 
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]

    ##### Label processing   ##############################################################
    # Target coly processing, Normalization process  , customize by model
    y_norm_fun = preprocess_pars.get('y_norm_fun', None)
    if y_norm_fun is not None:
        df[coly] = df[coly].apply(lambda x: y_norm_fun(x))

    ########### colnum procesing   #########################################################
    for x in colnum:
        df[x] = df[x].astype("float32")
    log(df[colall].dtypes)

    ### Map numerics to Category bin
    dfnum_bin, colnum_binmap = pd_colnum_tocat(df, colname=colnum, colexclude=None, colbinmap=None,
                                               bins=10, suffix="_bin", method="uniform",
                                               return_val="dataframe,param")
    log(colnum_binmap)
    save_features(dfnum_bin, 'dfnum_binmap', path_train_features )


    ### Renaming colunm_bin with suffix 
    colnum_bin = [x + "_bin" for x in list(colnum_binmap.keys())]
    log(colnum_bin)

    ### colnum bin to One Hot
    dfnum_hot, colnum_onehot = pd_col_to_onehot(dfnum_bin[colnum_bin], colname=colnum_bin,
                                                colonehot=None, return_val="dataframe,param")
    log(colnum_onehot)
    save_features(dfnum_hot, 'dfnum_onehot', path_train_features )

    ##### Colcat processing   ################################################################
    colcat_map = pd_colcat_mapping(df, colcat)
    log(df[colcat].dtypes, colcat_map)

    #### colcat to onehot
    dfcat_hot, colcat_onehot = pd_col_to_onehot(df[colcat], colname=colcat,
                                                colonehot=None, return_val="dataframe,param")
    log(dfcat_hot[colcat_onehot].head(5))
    save_features(dfcat_hot, 'dfcat_onehot', path_train_features )

    #### Colcat to integer encoding
    dfcat_bin, colcat_bin_map = pd_colcat_toint(df[colcat], colname=colcat,
                                                colcat_map=None, suffix="_int")
    colcat_bin = list(dfcat_bin.columns)
    save_features(dfcat_bin, 'dfcat_bin', path_train_features )

    ####### colcross cross features   ##############################################################
    df_onehot = dfcat_hot.join(dfnum_hot, on=colid, how='left')
    colcross_single_onehot_select = []
    for t in list(df_onehot) :
        for c1 in colcross_single :
            if c1 in t :
               colcross_single_onehot_select.append(t)
        

    df_onehot = df_onehot[colcross_single_onehot_select ]
    dfcross_hot, colcross_pair = pd_feature_generate_cross(df_onehot, colcross_single_onehot_select,
                                                           pct_threshold=0.02,
                                                           m_combination=2)
    log(dfcross_hot.head(2).T)
    colcross_pair_onehot = list(dfcross_hot.columns)
    save_features(dfcross_hot, 'dfcross_onehot', path_train_features )
    del df_onehot
    gc.collect()



    ################################################################################################
    ##### Save pre-processor meta-parameters
    os.makedirs(path_pipeline_export, exist_ok=True)
    log(path_pipeline_export)
    for t in ['colid',
              "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
              "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
              'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns

              "coly", "y_norm_fun"
              ]:
        tfile = f'{path_pipeline_export}/{t}.pkl'
        log(tfile)
        save(locals()[t], tfile)

    log("y_norm.pkl")
    save(y_norm_fun, f'{path_pipeline_export}/y_norm.pkl' )

   
    ######  Merge AlL  #############################################################################
    dfX = pd.concat((df[colnum], dfnum_bin, dfnum_hot,
                     dfcat_bin, dfcat_hot,
                     dfcross_hot,
                     df[coly]
                     ), axis=1)
    save_features(dfX, 'dfX', path_train_features )

    colX = list(dfX.columns)
    colX.remove(coly)

    cols_family = {
        'colid': colid,    'coly': coly, 'colall': colall,
        'colnum': colnum,
        'colnum_bin': colnum_bin,
        'colnum_onehot': colnum_onehot,
        
        'colcat_bin': colcat_bin,
        'colcat_onehot': colcat_onehot,

        'colcross_single_onehot_select' : colcross_single_onehot_select
       ,'colcross_pair_onehot':           colcross_pair_onehot
        #'colcross_onehot': colcross_onehot,
    }
    return dfX, cols_family


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
              run_preprocess=1, ):
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
                                 preprocess_pars, filter_pars)
    model_dict['data_pars']['coly'] = cols['coly']
    
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
    dfXy.to_parquet(path_check_out + "/dfX.parquet")  # train input data
    dfXytest.to_parquet(path_check_out + "/dfXtest.parquet")  # Test input data
    log("######### finish #################################", )


if __name__ == "__main__":
    import fire
    fire.Fire()




