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
import os
import json
import importlib


####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")

#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


####################################################################################################
####################################################################################################
def log(*s, n=0, m=0):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)

from util_feature import   load, save_list, load_function_uri, save



####################################################################################################
####################################################################################################
def save_features(df, name, path):
    if path is not None :
       os.makedirs( f"{path}/{name}", exist_ok=True)
       df.to_parquet( f"{path}/{name}/features.parquet")


from run_preprocess import  preprocess, preprocess_load


####################################################################################################
##### train    #####################################################################################
def map_model(model_name):
    """

    :param model_name:
    :return:
    """
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

    :param model_dict:
    :param dfX:
    :param cols_family:
    :param post_process_fun:
    :return:
    """
    model_pars, compute_pars = model_dict['model_pars'], model_dict['compute_pars']
    data_pars = model_dict['data_pars']
    model_name, model_path = model_pars['config_model_name'], model_pars['model_path']
    metric_list = compute_pars['metric_list']

    log("#### Data preparation #############################################################")
    log(dfX.shape)
    dfX    = dfX.sample(frac=1.0)
    itrain = int(0.6 * len(dfX))
    ival   = int(0.8 * len(dfX))
    colid  = cols_family['colid']
    colsX  = data_pars['cols_model']
    coly   = data_pars['coly']
    log("dfX", dfX[colsX] )
    log("dfy", dfX[coly] )

    data_pars['data_type'] = 'ram'
    data_pars['train'] = {'Xtrain' : dfX[colsX].iloc[:itrain, :],
                          'ytrain' : dfX[coly].iloc[:itrain],
                          'Xtest'  : dfX[colsX].iloc[itrain:ival, :],
                          'ytest'  : dfX[coly].iloc[itrain:ival],

                          'Xval'   : dfX[colsX].iloc[ival:, :],
                          'yval'   : dfX[coly].iloc[ival:],
                          }



    log("#### Model Instance ##########################################################")
    # from config_model import map_model    
    modelx = map_model(model_name)    
    log(modelx)
    modelx.reset()
    modelx.init(model_pars, compute_pars=compute_pars)
    modelx.fit(data_pars, compute_pars)

    log("#### Metrics #################################################################")
    from util_feature import  metrics_eval

    stats               = {}
    ypred, ypred_proba  = modelx.predict(dfX[colsX], compute_pars=compute_pars)
    dfX[coly + '_pred'] = ypred  # y_norm(ypred, inverse=True)

    dfX[coly]            = dfX[coly].apply(lambda  x : post_process_fun(x) )
    dfX[coly + '_pred']  = dfX[coly + '_pred'].apply(lambda  x : post_process_fun(x) )


    if ypred_proba is None :
       ypred_proba_val = None

    elif len(ypred_proba.shape) <= 1:
       ypred_proba_val      = ypred_proba[ival:]
       dfX[coly + '_proba'] = ypred_proba

    elif len(ypred_proba.shape) > 1:
        from util_feature import np_conv_to_one_col
        ypred_proba_val      = ypred_proba[ival:,:]
        dfX[coly + '_proba'] = np_conv_to_one_col(ypred_proba, ";")  ### merge into string "p1,p2,p3,p4"
        log(dfX.head(3).T)

    log("Actual    : ",  dfX[coly ])
    log("Prediction: ",  dfX[coly + '_pred'])

    metrics_test = metrics_eval(metric_list,
                                ytrue       = dfX[coly].iloc[ival:],
                                ypred       = dfX[coly + '_pred'].iloc[ival:],
                                ypred_proba = ypred_proba_val )
    stats['metrics_test'] = metrics_test
    log(stats)


    log("############ saving model, dfX, columns", model_path)
    os.makedirs(model_path, exist_ok=True)
    modelx.save(model_path, stats)
    save(colsX, model_path + "/colsX.pkl")
    save(coly,  model_path + "/coly.pkl")


    log(modelx.model.model_pars, modelx.model.compute_pars)
    a = load(model_path + "/model.pkl")
    log("############ check re-loaded", a.model_pars)
    
    return dfX.iloc[:ival, :].reset_index(), dfX.iloc[ival:, :].reset_index()


####################################################################################################
############CLI Command ############################################################################
def run_train(config_model_name, path_data, path_output, path_config_model="source/config_model.py", n_sample=5000,
              run_preprocess=1,  mode="run_preprocess"):
    """
      Configuration of the model is in config_model.py file
    :param config_model_name:
    :param path_data:
    :param path_output:
    :param path_config_model:
    :param n_sample:
    :param run_preprocess:
    :param mode:
    :return:
    """
    path_output       = root + path_output
    path_data         = root + path_data
    path_pipeline_out = path_output   + "/pipeline/"
    path_model_out    = path_output   + "/model/"
    path_check_out    = path_output   + "/check/"
    path_train_X      = path_data     + "/features.zip"#.zip
    path_train_y      = path_data     + "/target.zip"#.zip
    path_features_store = path_output + '/features_store/'  #path_data replaced with path_output, because preprocessed files are stored there
    log(path_output)

    log("#### Model Params Dynamic loading  ###############################################")
    model_dict_fun = load_function_uri(uri_name=path_config_model + "::" + config_model_name)
    model_dict     = model_dict_fun(path_model_out)   ### params
    log( model_dict )

    log("#### load input column family  ###################################################")
    try :
        cols_group = model_dict['data_pars']['cols_input_type']  ### the model config file
    except :
        cols_group = json.load(open(path_data + "/cols_group.json", mode='r'))
    log(cols_group)


    log("#### Preprocess  #################################################################")
    preprocess_pars = model_dict['model_pars']['pre_process_pars']
    filter_pars     = model_dict['data_pars']['filter_pars']
     
    if mode == "run_preprocess" :
        dfXy, cols      = preprocess(path_train_X, path_train_y, path_pipeline_out, cols_group, n_sample,
                                 preprocess_pars, filter_pars, path_features_store=path_features_store)

    elif mode == "load_preprocess" :  #### Load existing data
        dfXy, cols      = preprocess_load(path_train_X, path_train_y, path_pipeline_out, cols_group, n_sample,
                                 preprocess_pars, filter_pars, path_features_store=path_features_store)

    ### Actual column for label
    model_dict['data_pars']['coly'] = cols['coly']

    
    ### Get actual column names from colum groups : colnum , colcat
    model_dict['data_pars']['cols_model'] = sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , [])                
    log("used columns",  model_dict['data_pars']['cols_model'] , model_dict['data_pars']['coly'])
    
   
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



def run_data_check(path_output, scoring):
    import pandas as pd
    try :
        #### Load model
        from source.util_feature import load
        from source.models import model_sklearn as modelx
        import sys
        from source import models
        sys.modules['models'] = models

        dir_model    = path_output
        modelx.model = load( dir_model + "/model/model.pkl" )
        stats        = load( dir_model + "/model/info.pkl" )
        colsX        = load( dir_model + "/model/colsX.pkl"   )
        coly         = load( dir_model + "/model/coly.pkl"   )
        print(stats)
        print(modelx.model.model)

        ### Metrics on test data
        log(stats['metrics_test'])

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
        from util_feature import  feature_importance_perm
        lgb_featimpt_train,_ = feature_importance_perm(modelx, dfX[colused], dfy,
                                                       colused,
                                                       n_repeats=1,
                                                       scoring=scoring)

        print(lgb_featimpt_train)
    except :
        pass


if __name__ == "__main__":
    import fire
    fire.Fire()




