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
    #pipe_default    = [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]

    pipe_list = [
      {'uri' : 'source/preprocessors.py::pdf_coly',               'pars': {}, 'cols_family': 'coly',      'df_group':'coly',         'type': 'coly' },
      {'uri' : 'source/preprocessors.py::pd_colnum_bin',          'pars': {}, 'cols_family': 'colnum',    'df_group':'dfnum_bin',    'type': '' },
      {'uri' : 'source/preprocessors.py::pd_colnum_binto_onehot', 'pars': {}, 'cols_family': 'dfnum_bin', 'df_group':'dfnum_onehot', 'type': '' },
      {'uri':  'source/preprocessors.py::pd_colcat_bin',          'pars': {}, 'cols_family': 'colcat',    'df_group':'dfcat_bin',    'type': ''},
      {'uri':  'source/preprocessors.py::pd_colcat_to_onehot',    'pars': {}, 'cols_family': 'dfcat_bin', 'df_group':'dfcat_onehot', 'type': ''},
      {'uri' : 'source/preprocessors.py::pd_colcross',            'pars': {}, 'cols_family': 'colcross',  'df_group':'dfcross_hot',  'type': 'cross' }
    ]

    # pipe_list    = preprocess_pars.get('pipe_list', pipe_default)
    pipe_list_X    = [ task for task in pipe_list  if task.get('type', '')  not in ['coly', 'cross']  ]
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
    print('--------------pipe_list-----------------')
    print(pipe_list)


    from _collections import OrderedDict
    dfi_all          =  OrderedDict() ### Dict of all features
    cols_family_full =  OrderedDict()


    if len(pipe_filter) > 0 :
        log("#####  Filter  #########################################################################")
        pipe_i       = pipe_list[ 0 ]
        pipe_fun     = load_function_uri(pipe_i['uri'])
        df, col_pars = pipe_fun(df, list(df.columns), pars=pipe_i.get('pars', {}))


    if len(pipe_list_y) > 0 :
       log("#####  coly  ###########################################################################")
       pipe_i       = pipe_list[ 0 ]
       pipe_fun     = load_function_uri(pipe_i['uri'])

       pars                        = pipe_i.get('pars', {})
       pars['path_features_store'] = path_features_store
       df, col_pars                = pipe_fun(df, cols_group['coly'], pars=pars)
       dfi_all['coly']             = df[cols_group['coly'] ]
       save_features(df[cols_group['coly'] ], "coly", path_features_store)  ### already saved



    #####  Processors  ######################################################################
    for pipe_i in pipe_list_X :
       log("###################", pipe_i, "##################################################")
       pipe_fun    = load_function_uri(pipe_i['uri'])    ### Load the code definition  into pipe_fun
       cols_name   = pipe_i['cols_family']
       df_group    = pipe_i['df_group']
       col_type    = pipe_i['type']
       try:
           cols_list = cols_group[cols_name]
           df_       = df[ cols_list]

       except:
           print(dfi_all[cols_name].columns)
           cols_list = list(dfi_all[cols_name].columns)
           df_       = dfi_all[cols_name]
       # cols_family = {}
       print(cols_list)
       cols_family     = {} # []  #{}
       flag_col_in_dfi = False
       if cols_name in dfi_all.keys():
           flag_col_in_dfi = True

       pars = pipe_i.get('pars', {})
       pars['path_features_store'] = path_features_store
       if col_type == 'cross':
           pars['dfnum_hot'] = dfi_all['dfnum_hot']  ### dfnum_hot --> dfcross
           pars['dfcat_hot'] = dfi_all['dfcat_hot']
           pars['colid'] = colid
           pars['colcross_single'] = colcross_single
           dfi, col_pars = pipe_fun(df_, cols_list, pars=pipe_i.get('pars', {}))
           ### Save on Disk column names ( pre-processor meta-params)  + dataframe intermediate
           cols_family[df_group] = list(dfi.columns)
           # cols_family.extend( list(dfi.columns) )  ### all columns names are unique !!!!

           save_features(dfi, df_group , path_features_store)  ### already saved
           ### Merge sub-family
           dfi_all[df_group] = pd.concat((dfi_all[df_group], dfi), axis=1) if dfi_all.get(df_group) is not None else dfi


       else:
           for cols_num, cols_i in enumerate(cols_list) :
                print('------------cols_i----------------')
                print(cols_i)
                dfi, col_pars = pipe_fun(df_[[cols_i]], [cols_i], pars=pipe_i.get('pars', {}))
                print('------------dfi.columns----------------')
                print(dfi.columns)
                print('------------dfi----------------')
                print(dfi)
                print('------------col_pars----------------')
                print(col_pars)
                ### Save on Disk column names ( pre-processor meta-params)  + dataframe intermediate
                cols_family[cols_i ]=list(dfi.columns)
                #cols_family.extend( list(dfi.columns) )  ### all columns names are unique !!!!

                save_features(dfi, df_group + "-" + cols_i, path_features_store)  ### already saved

                ### Merge sub-family
                dfi_all[df_group] =  pd.concat((dfi_all[df_group], dfi), axis=1)  if dfi_all.get(df_group) is not None else dfi

       print('------------dfi_all-----------------')
       print(dfi_all)
       print('------------cols_family-------------')
       print(cols_family)
       ### Flatten the columns
       cols_family_export          = []#
       # for _, col_list in cols_family.items():
       #     for coli in col_list:
       #         cols_family_export.append(coli)
       cols_family_full[cols_name] = cols_family
       print(cols_family_full)

       ### save on disk
       save(cols_family_export, f'{path_pipeline_export}/{cols_name}.pkl')
       save_features(dfi_all[df_group], cols_name, path_features_store)
       # log(dfi_all.head(6))


    ######  Merge AlL  #################################################################
    dfXy = df[colnum + colcat]
    #for t in [ 'dfnum_bin', 'dfnum_hot', 'dfcat_bin', 'dfcat_hot', 'dfcross_hot',
    #           'dfdate',  'dftext'  ] :
    #    if t in dfi_all:

    for t in dfi_all.keys():
        dfXy = pd.concat((dfXy, dfi_all[t] ), axis=1)
    print('----------dfXy----------')
    print(dfXy)
    print('----------dfXy.columns----------')
    print(dfXy.columns)
    save_features(dfXy, 'dfX', path_features_store)
    # print(cols_family)
    colXy = list(dfXy.columns)
    colXy.remove(coly)    ##### Only X columns
    cols_family_full['colX'] = colXy
    save(colXy,            f'{path_pipeline_export}/colsX.pkl' )
    save(cols_family_full, f'{path_pipeline_export}/cols_family.pkl' )


    ###### Return values  ##############################################################
    return dfXy, cols_family_full




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






"""
    ##### Save pre-processor meta-paramete
    os.makedirs(path_pipeline_export, exist_ok=True)
    log(path_pipeline_export)
    cols_family = {}

    for t in ['colid',
              "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
              "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
              'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns

              'coldate',
              'coltext',

              "coly", "y_norm_fun"
              ]:
        tfile = f'{path_pipeline_export}/{t}.pkl'
        log(tfile)
        t_val = locals().get(t, None)
        if t_val is not None :
           save(t_val, tfile)
           cols_family[t] = t_val
"""





