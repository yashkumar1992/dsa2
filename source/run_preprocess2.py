# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
cd analysis
 run preprocess
"""
import warnings
warnings.filterwarnings('ignore')
import sys
import gc
import os
import pandas as pd
import json, copy
from pprint import pprint


####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)



####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)


def log_pd(df, *s, n=0, m=1):
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump,  df.head(n), flush=True)


from util_feature import  save, load_function_uri



####################################################################################################
####################################################################################################
from util_feature import  load_dataset


def save_features(df, name, path=None):
    """
    :param df:
    :param name:
    :param path:
    :return:
    """
    if path is not None :
       os.makedirs( f"{path}/{name}" , exist_ok=True)
       if isinstance(df, pd.Series):
           df0=df.to_frame()
       else:
           df0=df
       log( f"{path}/{name}/features.parquet" )
       df0.to_parquet( f"{path}/{name}/features.parquet")
    else:
       log("No saved features, path is none")


def load_features(name, path):
    try:
        return pd.read_parquet(f"{path}/{name}/features.parquet")
    except:
        log("Not available", path, name)
        return None


####################################################################################################
####################################################################################################
def preprocess(path_train_X="", path_train_y="", path_pipeline_export="", cols_group=None, n_sample=5000,
               preprocess_pars={}, filter_pars={}, path_features_store=None):
    """
    :param path_train_X:
    :param path_train_y:
    :param path_pipeline_export:
    :param cols_group:
    :param n_sample:
    :param preprocess_pars:
    :param filter_pars:
    :param path_features_store:
    :return:
    """
    from util_feature import (pd_colnum_tocat, pd_col_to_onehot, pd_colcat_mapping, pd_colcat_toint,
                              pd_feature_generate_cross)

    ##### column names for feature generation #####################################################
    log(cols_group)
    coly            = cols_group['coly']  # 'salary'
    colid           = cols_group['colid']  # "jobId"
    colcat          = cols_group['colcat']  # [ 'companyId', 'jobType', 'degree', 'major', 'industry' ]
    colnum          = cols_group['colnum']  # ['yearsExperience', 'milesFromMetropolis']

    colcross_single = cols_group.get('colcross', [])   ### List of single columns
    coltext         = cols_group.get('coltext', [])
    coldate         = cols_group.get('coldate', [])
    colall          = colnum + colcat + coltext + coldate
    log(colall)

    #### Pipeline Execution
    pipe_default = [
      {'uri' : 'source/preprocessors.py::pd_coly',                'pars': preprocess_pars, 'cols_family': 'coly',       'cols_out':'coly',         'type': 'coly' },

      {'uri' : 'source/preprocessors.py::pd_colnum_bin',          'pars': {}, 'cols_family': 'colnum',     'cols_out':'dfnum_bin',    'type': '' },
      {'uri' : 'source/preprocessors.py::pd_colnum_binto_onehot', 'pars': {}, 'cols_family': 'colnum_bin', 'cols_out':'dfnum_onehot', 'type': '' },
      {'uri':  'source/preprocessors.py::pd_colcat_bin',          'pars': {}, 'cols_family': 'colcat',     'cols_out':'dfcat_bin',    'type': ''},
      {'uri':  'source/preprocessors.py::pd_colcat_to_onehot',    'pars': {}, 'cols_family': 'colcat_bin', 'cols_out':'dfcat_onehot', 'type': ''},
      {'uri' : 'source/preprocessors.py::pd_colcross',            'pars': {}, 'cols_family': 'colcross',   'cols_out':'dfcross_hot',  'type': 'cross' }
    ]

    pipe_list = pipe_default
    # pipe_list    = preprocess_pars.get('pipe_list', pipe_default)
    pipe_list_X    = [ task for task in pipe_list  if task.get('type', '')  not in ['coly', 'filter']  ]
    pipe_list_y    = [ task for task in pipe_list  if task.get('type', '')   in ['coly']  ]
    pipe_filter    = [ task for task in pipe_list  if task.get('type', '')   in ['filter']  ]
    ##### Load data ###########################################################################
    df = load_dataset(path_train_X, path_train_y, colid, n_sample= n_sample)
    print(df)


    ##### Generate features ###################################################################
    os.makedirs(path_pipeline_export, exist_ok=True)
    log(path_pipeline_export)
    print('--------------cols_group-----------------')
    print(cols_group)
    print('--------------pipe_ist-----------------')
    print(pipe_list)


    from _collections import OrderedDict
    dfi_all          = {} ### Dict of all features
    cols_family_full = {}


    if len(pipe_filter) > 0 :
        log("#####  Filter  #########################################################################")
        pipe_i       = pipe_filter[ 0 ]
        pipe_fun     = load_function_uri(pipe_i['uri'])
        df, col_pars = pipe_fun(df, list(df.columns), pars=pipe_i.get('pars', {}))


    if len(pipe_list_y) > 0 :
        log("#####  coly  ###########################################################################")
        pipe_i       = pipe_list_y[ 0 ]
        pipe_fun     = load_function_uri(pipe_i['uri'])

        pars                        = pipe_i.get('pars', {})
        pars['path_features_store'] = path_features_store
        df, col_pars                = pipe_fun(df, cols_group['coly'], pars=pars)   ### coly can remove rows
        dfi_all['coly']             = df[cols_group['coly'] ]
        save_features(df[cols_group['coly'] ], "coly", path_features_store)  ### already saved
        cols_family_full['coly']    = cols_group['coly']


    #####  Processors  ######################################################################
    for pipe_i in pipe_list_X :
       log("###################", pipe_i, "#######################################################")
       pipe_fun    = load_function_uri(pipe_i['uri'])    ### Load the code definition  into pipe_fun
       cols_name   = pipe_i['cols_family']
       # df_group    = pipe_i['df_group']
       col_type    = pipe_i['type']

       try:
           cols_list = cols_group[cols_name]
           df_       = df[ cols_list]

       except: #### Previously computed
           cols_list = list(dfi_all[cols_name].columns)
           df_       = dfi_all[cols_name]
       print(cols_name, cols_list)

       cols_family     = {}
       pars            = pipe_i.get('pars', {})
       pars['path_features_store'] = path_features_store

       if col_type == 'cross':
           pars['dfnum_hot']       = dfi_all['colnum_onehot']  ### dfnum_hot --> dfcross
           pars['dfcat_hot']       = dfi_all['colcat_onehot']
           pars['colid']           = colid
           pars['colcross_single'] = cols_group.get('colcross', [])
           dfi, col_pars           = pipe_fun(df_, cols_list, pars= pars)
           print('--------------col_pars-----------------',col_pars['cols_new'].items())
           ### colnum, colnum_bin into cols_family
           for colname, colist in  col_pars['cols_new'].items() :
              cols_family_full[cols_name] =  cols_family_full.get(colname, []) + colist

           save_features(dfi, cols_name , path_features_store)  ### already saved
           ### Merge sub-family
           dfi_all[cols_name] = pd.concat((dfi_all[cols_name], dfi), axis=1) if cols_name in dfi_all else dfi

       else:
           for cols_i in cols_list :
                log('------------cols_i----------------', cols_i)
                dfi, col_pars = pipe_fun(df_[[cols_i]], [cols_i], pars= pars)
                print(dfi)
                print(col_pars)

                ### colnum, colnum_bin into cols_family_full
                for colj, colist in  col_pars['cols_new'].items() :
                  cols_family_full[colj] =  cols_family_full.get(colj, []) + colist
                  save(cols_family_full[colj], f'{path_pipeline_export}/{colj}.pkl')   ### Not Efficient

                  dfi_all[colj] =  pd.concat((dfi_all[colj], dfi), axis=1)  if colj in dfi_all else dfi
                  save_features(dfi_all[colj], colj, path_features_store)

       print('------------dfi_all---------------------', dfi_all)
       print('------------cols_family-----------------', cols_family)


    ######  Merge AlL int dfXy  ##################################################################
    dfXy = df[ [coly] + colnum + colcat ]
    for t in dfi_all.keys():
        if t not in [ 'coly', 'colnum', 'colcat'] :
           dfXy = pd.concat((dfXy, dfi_all[t] ), axis=1)

    log('----------dfXy------------------', dfXy)
    log('----------dfXy.columns----------', dfXy.columns)
    save_features(dfXy, 'dfX', path_features_store)


    colXy = list(dfXy.columns)
    colXy.remove(coly)    ##### Only X columns
    if len(colid)>0:
        cols_family_full['colid']=colid
    cols_family_full['colX'] = colXy
    save(colXy,            f'{path_pipeline_export}/colsX.pkl' )
    save(cols_family_full, f'{path_pipeline_export}/cols_family.pkl' )


    ###### Return values  #######################################################################
    print('cols_family_full')
    pprint(cols_family_full)
    return dfXy, cols_family_full





def preprocess_inference(df, path_pipeline="data/pipeline/pipe_01/", preprocess_pars={}, cols_group=None):
    """
      at inference time
    """
    from util_feature import load, load_function_uri, load_dataset
    from util_feature import (pd_colnum_tocat, pd_col_to_onehot, pd_colcat_toint,
                              pd_feature_generate_cross)


    #### Pipeline Execution  ####################################################
    pipe_default = [

      #{'uri' : 'source/preprocessors.py::pd_colnum_bin',          'pars': {}, 'cols_family': 'colnum',     'cols_out':'dfnum_bin',    'type': '' },
      #{'uri' : 'source/preprocessors.py::pd_colnum_binto_onehot', 'pars': {}, 'cols_family': 'colnum_bin', 'cols_out':'dfnum_onehot', 'type': '' },
      {'uri':  'source/preprocessors.py::pd_colcat_bin',          'pars': {}, 'cols_family': 'colcat',     'cols_out':'dfcat_bin',    'type': ''},
      #{'uri':  'source/preprocessors.py::pd_colcat_to_onehot',    'pars': {}, 'cols_family': 'colcat_bin', 'cols_out':'dfcat_onehot', 'type': ''},
      #{'uri' : 'source/preprocessors.py::pd_colcross',            'pars': {}, 'cols_family': 'colcross',   'cols_out':'dfcross_hot',  'type': 'cross' }
    ]

    pipe_list = pipe_default
    # pipe_list    = preprocess_pars.get('pipe_list', pipe_default)
    pipe_list_X    = [ task for task in pipe_list  if task.get('type', '')  not in ['coly', 'filter']  ]
    pipe_filter    = [ task for task in pipe_list  if task.get('type', '')   in ['filter']  ]
    ##### Load data ################################################################################
    print(df)


    ##### column names for feature generation #####################################################
    log(cols_group)   ### list of model columns familty
    colid           = cols_group['colid']   # "jobId"
    colcat          = cols_group['colcat']  # [ 'companyId', 'jobType', 'degree', 'major', 'industry' ]
    colnum          = cols_group['colnum']  # ['yearsExperience', 'milesFromMetropolis']

    colall          = colnum + colcat
    log(colall)


    ##### Generate features ########################################################################
    print('--------------pipe_ist-----------------', pipe_list)
    dfi_all          = {} ### Dict of all features
    cols_family_full = {}


    if len(pipe_filter) > 0 :
        log("#####  Filter  #######################################################################")
        pipe_i       = pipe_filter[ 0 ]
        pipe_fun     = load_function_uri(pipe_i['uri'])
        df, col_pars = pipe_fun(df, list(df.columns), pars=pipe_i.get('pars', {}))


    #####  Processors  ######################################################################
    for pipe_i in pipe_list_X :
       log("###################", pipe_i, "#######################################################")
       pipe_fun    = load_function_uri(pipe_i['uri'])    ### Load the code definition  into pipe_fun
       cols_name   = pipe_i['cols_family']
       col_type    = pipe_i['type']

       try:
           cols_list = cols_group[cols_name]
           df_       = df[ cols_list]

       except: #### Previously computed
           cols_list = list(dfi_all[cols_name].columns)
           df_       = dfi_all[cols_name]
       print(cols_name, cols_list)

       cols_family     = {}
       pars            = pipe_i.get('pars', {})
       pars['path_pipeline'] = path_pipeline   ### Storage of local data.

       if col_type == 'cross':
           pars['dfnum_hot']       = dfi_all['colnum_onehot']  ### dfnum_hot --> dfcross
           pars['dfcat_hot']       = dfi_all['colcat_onehot']
           pars['colid']           = colid
           pars['colcross_single'] = cols_group.get('colcross', [])
           dfi, col_pars           = pipe_fun(df_, cols_list, pars= pars)
           print('--------------col_pars-----------------',col_pars['cols_new'].items())
           ### colnum, colnum_bin into cols_family
           for colname, colist in  col_pars['cols_new'].items() :
              cols_family_full[cols_name] =  cols_family_full.get(colname, []) + colist

           ### Merge sub-family
           dfi_all[cols_name] = pd.concat((dfi_all[cols_name], dfi), axis=1) if cols_name in dfi_all else dfi

       else:
           for cols_i in cols_list :
                log('------------cols_i----------------', cols_i)
                dfi, col_pars = pipe_fun(df_[[cols_i]], [cols_i], pars= pars)
                print(dfi)
                print(col_pars)

                ### colnum, colnum_bin into cols_family_full
                for colj, colist in  col_pars['cols_new'].items() :
                  cols_family_full[colj] =  cols_family_full.get(colj, []) + colist
                  dfi_all[colj] =  pd.concat((dfi_all[colj], dfi), axis=1)  if colj in dfi_all else dfi

       print('------------dfi_all---------------------', dfi_all)
       print('------------cols_family-----------------', cols_family)


    ######  Merge AlL int dfXy  #################################################################
    dfXy = df[  colnum + colcat ]
    for t in dfi_all.keys():
        if t not in [  'colnum', 'colcat'] :
           dfXy = pd.concat((dfXy, dfi_all[t] ), axis=1)

    log('----------dfXy------------------', dfXy, dfXy.columns)

    colXy = list(dfXy.columns)
    if len(colid)>0:
        cols_family_full['colid']=colid
    cols_family_full['colX'] = colXy


    ###### Return values  #######################################################################
    print('cols_family_full')
    pprint(cols_family_full)
    return dfXy, cols_family_full



def preprocess_inference_old(df, path_pipeline="data/pipeline/pipe_01/", preprocess_pars={}):
    """
      at inference time
    """
    from util_feature import load, load_function_uri, load_dataset
    from util_feature import (pd_colnum_tocat, pd_col_to_onehot, pd_colcat_toint,
                              pd_feature_generate_cross)

    log("########### Load column by column type ##################################")
    colid          = load(f'{path_pipeline}/colid.pkl')
    coly           = load(f'{path_pipeline}/coly.pkl')
    colcat         = load(f'{path_pipeline}/colcat.pkl')
    colcat_onehot  = load(f'{path_pipeline}/colcat_onehot.pkl')
    colcat_bin_map = load(f'{path_pipeline}/colcat_bin_map.pkl')

    colnum         = load(f'{path_pipeline}/colnum.pkl')
    colnum_binmap  = load(f'{path_pipeline}/colnum_binmap.pkl')
    colnum_onehot  = load(f'{path_pipeline}/colnum_onehot.pkl')

    ### OneHot column selected for cross features
    colcross_single_onehot_select = load(f'{path_pipeline}/colcross_single_onehot_select.pkl')

    pipe_default    = [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
    pipe_list       = preprocess_pars.get('pipe_list', pipe_default)


    if "dfcat_hot" in pipe_list :
       log("###### Colcat to onehot ###############################################")
       dfcat_hot, _ = pd_col_to_onehot(df[colcat],  colname=colcat,
                                                  colonehot=colcat_onehot, return_val="dataframe,param")
       log(dfcat_hot[colcat_onehot].head(5))


    if "dfcat_bin" in pipe_list :
        log("###### Colcat as integer encoded  ####################################")
        dfcat_bin, _ = pd_colcat_toint(df[colcat],  colname=colcat,
                                                    colcat_map=colcat_bin_map, suffix="_int")
        colcat_bin = list(dfcat_bin.columns)

    if "dfnum_bin" in pipe_list :
        log("###### Colnum Preprocess   ###########################################")
        dfnum_bin, _ = pd_colnum_tocat(df, colname=colnum, colexclude=None,
                                         colbinmap=colnum_binmap,
                                         bins=-1, suffix="_bin", method="",
                                         return_val="dataframe,param")
        log(colnum_binmap)
        colnum_bin = [x + "_bin" for x in list(colnum_binmap.keys())]
        log(dfnum_bin[colnum_bin].head(5))


    if "dfnum_hot" in pipe_list :
        ###### Map numerics bin to One Hot
        dfnum_hot, _ = pd_col_to_onehot(dfnum_bin[colnum_bin], colname=colnum_bin,
                                         colonehot=colnum_onehot, return_val="dataframe,param")
        log(dfnum_hot[colnum_onehot].head(5))


    if "dfcross_hot" in pipe_list :
        log("####### colcross cross features   ###################################################")
        dfcross_hot = pd.DataFrame()
        if colcross_single_onehot_select is not None :
            df_onehot = dfcat_hot.join(dfnum_hot, on=colid, how='left')

            # colcat_onehot2 = [x for x in colcat_onehot if 'companyId' not in x]
            # log(colcat_onehot2)
            # colcross_single = colnum_onehot + colcat_onehot2
            df_onehot = df_onehot[colcross_single_onehot_select]
            dfcross_hot, colcross_pair = pd_feature_generate_cross(df_onehot, colcross_single_onehot_select,
                                                                    pct_threshold=0.02,
                                                                    m_combination=2)
            log(dfcross_hot.head(2).T)
            colcross_onehot = list(dfcross_hot.columns)
            del df_onehot ;    gc.collect()


    log("##### Merge data type together  :   #######################3############################ ")
    dfX = df[ colnum + colcat ]
    for t in [ 'dfnum_bin', 'dfnum_hot', 'dfcat_bin', 'dfcat_hot', 'dfcross_hot',   ] :
        if t in locals() :
           dfX = pd.concat((dfX, locals()[t] ), axis=1)
           # log(t, list(dfX.columns))

    colX = list(dfX.columns)
    #colX.remove(coly)
    del df ;    gc.collect()


    log("###### Export columns group   ##########################################################")
    cols_family = {}
    for t in ['colid','coly', #added 'coly'
              "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
              "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
              'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
              'colsX', 'coly'
              ]:
        t_val = locals().get(t, None)
        if t_val is not None :
           cols_family[t] = t_val



    return dfX, cols_family



def preprocess_load(path_train_X="", path_train_y="", path_pipeline_export="", cols_group=None, n_sample=5000,
               preprocess_pars={}, filter_pars={}, path_features_store=None):

    from source.util_feature import load

    dfXy        = pd.read_parquet(path_features_store + "/dfX/features.parquet")

    try :
       dfy  = pd.read_parquet(path_features_store + "/dfy/features.parquet")
       dfXy = dfXy.join(dfy, on= cols_group['colid']  , how="left")
    except :
       log('Error no label', path_features_store + "/dfy/features.parquet")

    cols_family = load(f'{path_pipeline_export}/cols_family.pkl')

    return  dfXy, cols_family


####################################################################################################
############CLI Command ############################################################################
def run_preprocess(model_name, path_data, path_output, path_config_model="source/config_model.py", n_sample=5000,
              mode='run_preprocess', path_features_store=None):     #prefix "pre" added, in order to make if loop possible
    """
      Configuration of the model is in config_model.py file
    """
    path_output         = root + path_output
    path_data           = root + path_data
    path_features_store = path_output + "/features_store/"
    path_pipeline_out   = path_output + "/pipeline/"
    path_model_out      = path_output + "/model/"
    path_check_out      = path_output + "/check/"
    path_train_X        = path_data   + "/features*"    ### Can be a list of zip or parquet files
    path_train_y        = path_data   + "/target*"      ### Can be a list of zip or parquet files
    log(path_output)


    # log("#### load input column family  ###################################################")
    # cols_group = json.load(open(path_data + "/cols_group.json", mode='r'))
    # log(cols_group)


    log("#### Model parameters Dynamic loading  ############################################")
    model_dict_fun = load_function_uri(uri_name= path_config_model + "::" + model_name)
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

    if mode == "run_preprocess" :
        dfXy, cols      = preprocess(path_train_X, path_train_y, path_pipeline_out, cols_group, n_sample,
                                 preprocess_pars, filter_pars, path_features_store)

    elif mode == "load_preprocess" :
        dfXy, cols      = preprocess_load(path_train_X, path_train_y, path_pipeline_out, cols_group, n_sample,
                                 preprocess_pars, filter_pars, path_features_store)
    print(cols)
    print('ss')
    model_dict['data_pars']['coly'] = cols['coly']

    ### Generate actual column names from colum groups : colnum , colcat
    model_dict['data_pars']['cols_model'] = sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , [])
    log(  model_dict['data_pars']['cols_model'] , model_dict['data_pars']['coly'])


    log("######### finish #################################", )


if __name__ == "__main__":
    import fire
    fire.Fire()









