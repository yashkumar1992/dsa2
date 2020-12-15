# -*- coding: utf-8 -*- 
"""

 ! activate py36 && python source/run_inference.py  run_predict  --n_sample 1000  --config_model_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data /data/input/train/
 

"""
import warnings
warnings.filterwarnings('ignore')
import sys
import gc
import os
import pandas as pd
import importlib

#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")
import util_feature


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement Logging
    print(sjump, sspace, s, sspace, flush=True)


from util_feature import load, load_function_uri
from util_feature import load_dataset


####################################################################################################
####################################################################################################
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


def predict(model_name, path_model, dfX, cols_family):
    """
    if config_model_name in ['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
        from models import model_sklearn as modelx

    elif config_model_name == 'model_bayesian_pyro':
        from models import model_bayesian_pyro as modelx

    elif config_model_name == 'model_widedeep':
        from models import model_widedeep as modelx
    """
    modelx = map_model(model_name)    
    modelx.reset()
    log(modelx, path_model)    
    #log(os.getcwd())
    sys.path.append( root)    #### Needed due to import source error    
    

    log("#### Load model  ############################################")
    modelx.model = load(path_model + "/model/model.pkl")
    # stats = load(path_model + "/model/info.pkl")
    colsX       = load(path_model + "/model/colsX.pkl")   ## column name
    # coly  = load( path_model + "/model/coly.pkl"   )
    assert colsX is not None, "cannot load colsx, " + path_model
    assert modelx.model is not None, "cannot load modelx, " + path_model
    log("#### modelx\n", modelx.model.model)

    log("### Prediction  ############################################")
    dfX1  = dfX.reindex(columns=colsX)   #reindex included
    ypred = modelx.predict(dfX1)

    return ypred



####################################################################################################
############CLI Command ############################################################################
def run_predict(model_name, path_model, path_data, path_output,cols_group, n_sample=-1):
    path_output   = root + path_output
    path_data     = root + path_data + "/features.zip"#.zip
    path_model    = root + path_model
    path_pipeline = path_model + "/pipeline/"
    path_test_X   = path_data + "/features.zip"   #.zip #added path to testing features
    log(path_data, path_model, path_output)

    colid            = load(f'{path_pipeline}/colid.pkl')

    df               = load_dataset(path_data, path_data_y=None, colid=colid, n_sample=n_sample)


    from run_preprocess2 import preprocess_inference   as preprocess
    dfX, cols_family = preprocess(df, path_pipeline, preprocess_pars={'cols_group':cols_group})
    
    ypred, yproba    = predict(model_name, path_model, dfX, cols_family)


    log("############ Saving prediction", ypred.shape, path_output)
    os.makedirs(path_output, exist_ok=True)
    df[cols_family["coly"] + "_pred"]       = ypred
    if yproba is not None :
       df[cols_family["coly"] + "_pred_proba"] = yproba
    df.to_csv(f"{path_output}/prediction.csv")
    log(df.head(8))

    log("###########  Export Specific")
    df[cols_family["coly"]] = ypred
    df[[cols_family["coly"]]].to_csv(f"{path_output}/pred_only.csv")



def run_data_check(path_data, path_data_ref, path_model, path_output, sample_ratio=0.5):
    """
     Calcualata Dataset Shift before prediction.
    """
    from run_preprocess2 import preprocess_inference   as preprocess
    path_output   = root + path_output
    path_data     = root + path_data
    path_data_ref = root + path_data_ref
    path_pipeline = root + path_model + "/pipeline/"

    os.makedirs(path_output, exist_ok=True)

    colid          = load(f'{path_pipeline}/colid.pkl')

    df1 = load_dataset(path_data_ref,colid=colid)
    dfX1, cols_family1 = preprocess(df1, path_pipeline)

    df2 = load_dataset(path_data,colid=colid)
    dfX2, cols_family2 = preprocess(df2, path_pipeline)

    colsX       = cols_family1["colnum_bin"] + cols_family1["colcat_bin"]
    dfX1        = dfX1[colsX]
    dfX2        = dfX2[colsX]

    nsample     = int(min(len(dfX1), len(dfX2)) * sample_ratio)
    metrics_psi = util_feature.pd_stat_dataset_shift(dfX2, dfX1,
                                                     colsX, nsample=nsample, buckets=7, axis=0)
    metrics_psi.to_csv(f"{path_output}/prediction_features_metrics.csv")
    log(metrics_psi)



###########################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()
