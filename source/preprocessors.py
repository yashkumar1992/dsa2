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
import util_feature


def save_features(df, name, path):
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
       df0.to_parquet( f"{path}/{name}/features.parquet")


####################################################################################################
def coltext_stopwords(text, stopwords=None, sep=" "):
    tokens = text.split(sep)
    tokens = [t.strip() for t in tokens if t.strip() not in stopwords]
    return " ".join(tokens)


def pd_coltext_clean( df, col, stopwords= None , pars=None):
    import string, re
    ntoken= pars.get('n_token', 1)
    df      = df.fillna("")
    dftext = df
    log(dftext)
    log(col)
    list1 = []
    list1.append(col)

    # fromword = [ r"\b({w})\b".format(w=w)  for w in fromword    ]
    # print(fromword)
    for col_n in list1:
        dftext[col_n] = dftext[col_n].fillna("")
        dftext[col_n] = dftext[col_n].str.lower()
        dftext[col_n] = dftext[col_n].apply(lambda x: x.translate(string.punctuation))
        dftext[col_n] = dftext[col_n].apply(lambda x: x.translate(string.digits))
        dftext[col_n] = dftext[col_n].apply(lambda x: re.sub("[!@,#$+%*:()'-]", " ", x))
        dftext[col_n] = dftext[col_n].apply(lambda x: coltext_stopwords(x, stopwords=stopwords))
    return dftext

def pd_coltext_wordfreq(df, col, stopwords, ntoken=100):
    """
    :param df:
    :param coltext:  text where word frequency should be extracted
    :param nb_to_show:
    :return:
    """
    sep=" "
    coltext_freq = df[col].apply(lambda x: pd.value_counts(x.split(sep))).sum(axis=0).reset_index()
    coltext_freq.columns = ["word", "freq"]
    coltext_freq = coltext_freq.sort_values("freq", ascending=0)
    log(coltext_freq)

    word_tokeep  = coltext_freq["word"].values[:ntoken]
    word_tokeep  = [  t for t in word_tokeep if t not in stopwords   ]

    return coltext_freq, word_tokeep


def nlp_get_stopwords():
    import json
    import string
    stopwords = json.load(open("source/utils/stopwords_en.json") )["word"]
    stopwords = [ t for t in string.punctuation ] + stopwords
    stopwords = [ "", " ", ",", ".", "-", "*", '€', "+", "/" ] + stopwords
    stopwords =list(set( stopwords ))
    stopwords.sort()
    print( stopwords )
    stopwords = set(stopwords)
    return stopwords


def pd_coltext(df, col, pars={}):
    from utils import util_text, util_model
    stopwords = pars['stopwords']
    dftext                              = pd_coltext_clean(df, col, stopwords= stopwords , pars=pars)
    coltext_freq, word_tokeep           = pd_coltext_wordfreq(df, col, stopwords, ntoken=100)  ## nb of words to keep

    dftext_tdidf_dict, word_tokeep_dict = util_text.pd_coltext_tdidf( dftext, coltext= col,  word_minfreq= 1,
                                                            word_tokeep = word_tokeep ,
                                                            return_val  = "dataframe,param"  )
    log(word_tokeep_dict)
    ###  Dimesnion reduction for Sparse Matrix
    dftext_svd_list, svd_list = util_model.pd_dim_reduction(dftext_tdidf_dict,
                                                   colname        = None,
                                                   model_pretrain = None,
                                                   colprefix      = col + "_svd",
                                                   method         = "svd",  dimpca=2,  return_val="dataframe,param")
    return dftext_svd_list



##### Filtering / cleaning rows :   #########################################################
def pd_filter_rows(df, col, pars):

    coly = col
    filter_pars =  pars
    def isfloat(x):
        try :
            a= float(x)
            return 1
        except:
            return 0

    ymin, ymax = pars.get('ymin', -9999999999.0), filter_pars.get('ymax', 999999999.0)

    df['_isfloat'] = df[ coly ].apply(lambda x : isfloat(x),axis=1 )
    df = df[ df['_isfloat'] > 0 ]
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]
    del df['_isfloat']

    return df, col



##### Label processing   ##################################################################
def pd_label_clean(df, col, pars):
    path_features_store = pars['path_features_store']
    # path_pipeline_export = pars['path_pipeline_export']
    coly = col=[0]
    y_norm_fun = None
    # Target coly processing, Normalization process  , customize by model
    log("y_norm_fun preprocess_pars")
    y_norm_fun = pars.get('y_norm_fun', None)
    if y_norm_fun is not None:
        df[coly] = df[coly].apply(lambda x: y_norm_fun(x))
        # save(y_norm_fun, f'{path_pipeline_export}/y_norm.pkl' )
        save_features(df[coly], 'dfy', path_features_store)
    return df,coly


def pd_coly(df, col, pars):
    ##### Filtering / cleaning rows :   #########################################################
    coly=col
    def isfloat(x):
        try :
            a= float(x)
            return 1
        except:
            return 0
    df['_isfloat'] = df[ coly ].apply(lambda x : isfloat(x) )
    df             = df[ df['_isfloat'] > 0 ]
    del df['_isfloat']

    ymin, ymax = pars.get('ymin', -9999999999.0), pars.get('ymax', 999999999.0)
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]
    ##### Label processing   ####################################################################
    y_norm_fun = None
    # Target coly processing, Normalization process  , customize by model
    log("y_norm_fun preprocess_pars")
    path_features_store = pars['path_features_store']

    y_norm_fun = pars.get('y_norm_fun', None)
    if y_norm_fun is not None:
        df[coly] = df[coly].apply(lambda x: y_norm_fun(x))
        # save(y_norm_fun, f'{path_pipeline_export}/y_norm.pkl' )
        save_features(df[coly], 'dfy', path_features_store)
    return df,col


def pd_colnum(df, col, pars):
    colnum = col
    for x in colnum:
        df[x] = df[x].astype("float32")
    log(df.dtypes)


def pd_colnum_normalize(df, col, pars):
    log("### colnum normalize  ###############################################################")
    from util_feature import pd_colnum_normalize
    colnum = col
    path_features_store = pars['path_features_store']
    pars = { 'pipe_list': [ {'name': 'fillna', 'naval' : 0.0 }, {'name': 'minmax'} ]}
    dfnum_norm, colnum_norm = pd_colnum_normalize(df, colname=colnum,  pars=pars, suffix = "_norm",
                                                  return_val="dataframe,param")
    log(colnum_norm)
    save_features(dfnum_norm, 'dfnum_norm', path_features_store)
    return dfnum_norm, colnum_norm


def pd_colnum_bin(df, col, pars):
    from util_feature import  pd_colnum_tocat
    path_features_store = pars['path_features_store']
    colnum = col
    log("### colnum Map numerics to Category bin  ###########################################")
    print(colnum)
    dfnum_bin, colnum_binmap = pd_colnum_tocat(df, colname=colnum, colexclude=None, colbinmap=None,
                                               bins=10, suffix="_bin", method="uniform",
                                               return_val="dataframe,param")
    log(colnum_binmap)
    ### Renaming colunm_bin with suffix
    colnum_bin = [x + "_bin" for x in list(colnum_binmap.keys())]
    log(colnum_bin)
    save_features(dfnum_bin, 'colnum_bin' + "-" + str(col), path_features_store)

    col_pars = {}
    col_pars['colnumbin_map'] = colnum_binmap
    col_pars['cols_new'] = {
     'colnum'     :  col ,    ###list
     'colnum_bin' :  colnum_bin       ### list
    }
    return dfnum_bin, col_pars



def pd_colnum_binto_onehot(df, col, pars):
    assert isinstance(col, list) and isinstance(df, pd.DataFrame)

    from util_feature import  pd_col_to_onehot
    dfnum_bin = df[col]
    path_features_store = pars['path_features_store']
    colnum_bin = col
    log("### colnum bin to One Hot")
    dfnum_hot, colnum_onehot = pd_col_to_onehot(dfnum_bin[colnum_bin], colname=colnum_bin,
                                                colonehot=None, return_val="dataframe,param")
    log(colnum_onehot)
    save_features(dfnum_hot, 'colnum_onehot', path_features_store)

    col_pars = {}
    col_pars['colnum_onehot'] = colnum_onehot
    col_pars['cols_new'] = {
     # 'colnum'        :  col ,    ###list
     'colnum_onehot' :  colnum_onehot       ### list
    }

    return dfnum_hot, col_pars



def pd_colcat_to_onehot(df, col, pars):
    # dfbum_bin = df[col]
    if len(col)==1:

        colnew       = [col[0] + "_onehot"]
        df[ colnew ] =  df[col]
        col_pars     = {}
        col_pars['colcat_onehot'] = colnew
        col_pars['cols_new'] = {
         # 'colnum'        :  col ,    ###list
         'colcat_onehot'   :  colnew      ### list
        }
        return df[colnew], col_pars

    path_features_store = pars['path_features_store']
    colcat = col

    log("#### colcat to onehot")
    dfcat_hot, colcat_onehot = util_feature.pd_col_to_onehot(df[colcat], colname=colcat,
                                                colonehot=None, return_val="dataframe,param")
    log(dfcat_hot[colcat_onehot].head(5))
    save_features(dfcat_hot, 'colcat_onehot', path_features_store)

    col_pars = {}
    col_pars['colcat_onehot'] = colcat_onehot
    col_pars['cols_new'] = {
     # 'colnum'        :  col ,    ###list
     'colcat_onehot' :  colcat_onehot       ### list
    }

    print("ok ------------")
    return dfcat_hot, col_pars




from util_feature import load

def pd_colcat_bin(df, col=None, pars=None):
    # dfbum_bin = df[col]
    path_pipeline = pars.get('path_pipeline', False)
    if  path_pipeline:
       colcat         = load(f'{path_pipeline}/colcat.pkl')
       colcat_bin_map = load(f'{path_pipeline}/colcat_bin_map.pkl')
    else :
       colcat         = col
       colcat_bin_map = None


    log("#### Colcat to integer encoding ")
    dfcat_bin, colcat_bin_map = util_feature.pd_colcat_toint(df[colcat], colname=colcat,
                                                colcat_map=  colcat_bin_map ,
                                                suffix="_int")
    colcat_bin = list(dfcat_bin.columns)
    ##### Colcat processing   ################################################################
    colcat_map = util_feature.pd_colcat_mapping(df, colcat)
    log(df[colcat].dtypes, colcat_map)


    if pars.get('path_features_store', None) is not None :
       path_features_store = pars['path_features_store']
       save_features(dfcat_bin, 'dfcat_bin', path_features_store)


    col_pars = {}
    col_pars['colcat_bin_map'] = colcat_bin_map
    col_pars['cols_new'] = {
     'colcat'     :  col ,    ###list
     'colcat_bin' :  colcat_bin       ### list
    }

    return dfcat_bin, col_pars



def pd_colcross(df, col, pars):
    log("#####  Cross Features From OneHot Features   ######################################")
    from util_feature import pd_feature_generate_cross


    path_features_store = pars['path_features_store']
    dfcat_hot = pars['dfcat_hot']
    dfnum_hot = pars['dfnum_hot']
    colid     = pars['colid']
    colcross_single   = pars['colcross_single']

    try :
       df_onehot = dfcat_hot.join(dfnum_hot, on=colid, how='left')
    except :
       df_onehot = copy.deepcopy(dfcat_hot)

    colcross_single_onehot_select = []
    for t in list(df_onehot) :
        for c1 in colcross_single :
            if c1 in t :
               colcross_single_onehot_select.append(t)

    df_onehot = df_onehot[colcross_single_onehot_select ]
    dfcross_hot, colcross_pair = pd_feature_generate_cross(df_onehot, colcross_single_onehot_select,
                                                           pct_threshold=0.02,  m_combination=2)
    log(dfcross_hot.head(2).T)
    colcross_pair_onehot = list(dfcross_hot.columns)
    save_features(dfcross_hot, 'colcross_onehot', path_features_store)
    del df_onehot ; gc.collect()


    col_pars = {}
    col_pars['colcat_bin_map'] = colcross_pair
    col_pars['cols_new'] = {
     # 'colcat'     :  col ,    ###list
     'colcat_bin' :  colcross_pair       ### list
    }

    return dfcross_hot, col_pars


def pd_coldate(df, col, pars):
    log("##### Coldate processing   #############################################################")
    from utils import util_date
    coldate = col
    path_features_store = pars['path_features_store']

    dfdate = None
    for coldate_i in coldate :
        dfdate_i =  util_date.pd_datestring_split( df[[coldate_i]] , coldate_i, fmt="auto", return_val= "split" )
        dfdate  = pd.concat((dfdate, dfdate_i))  if dfdate is not None else dfdate_i
        save_features(dfdate_i, 'dfdate_' + coldate_i, path_features_store)
    save_features(dfdate, 'dfdate', path_features_store)
    return dfdate, None





if __name__ == "__main__":
    import fire
    fire.Fire()




"""

  coltext ---> coltext-coli-svd
  
  coldate ---> coltext-coli



    log("#### Data preparation #############################################################")
    log(dfX.shape)
    dfX    = dfX.sample(frac=1.0)
    itrain = int(0.6 * len(dfX))
    ival   = int(0.8 * len(dfX))
    colid  = cols_family['colid']
    colsX  = data_pars['cols_model']
    coly   = data_pars['coly']
    print('colsX',colsX)
    rm=["name", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access", "interaction", "house_rules", "host_name", "host_about", "amenities"]
    colsX = list(set(colsX) - set(rm))

    for col in rm:
        col1=col+'_svd_0'
        col2=col+'_svd_1'
        colsX.append(col1)
        colsX.append(col2)
    rm1=["last_review", "host_since", "first_review", "last_scraped"]
    colsX = list(set(colsX) - set(rm1))
    for col in rm1:
        col1=col+'_year'
        col2=col+'_month'
        col3=col+'_day'
        colsX.append(col1)
        colsX.append(col2)
        colsX.append(col3)
    dfX.fillna(0)
    data_pars['data_type'] = 'ram'
    data_pars['train'] = {'Xtrain' : dfX[colsX].iloc[:itrain, :],
                          'ytrain' : dfX[coly].iloc[:itrain],
                          'Xtest'  : dfX[colsX].iloc[itrain:ival, :],
                          'ytest'  : dfX[coly].iloc[itrain:ival],

                          'Xval'   : dfX[colsX].iloc[ival:, :],
                          'yval'   : dfX[coly].iloc[ival:],
                          }
                          
                          


"""



"""
def pd_coltext(df, col, pars):
    log("##### Coltext processing   ###############################################################")
    path_features_store = pars['path_features_store']
    coltext = col

    stopwords = nlp_get_stopwords()
    pars      = {'n_token' : 100 , 'stopwords': stopwords}
    dftext    = None
    for coltext_i in coltext :
        ##### Run the text processor on each column text  #############################
        dftext_i = pipe_text( df[[coltext_i ]], coltext_i, pars )
        dftext   = pd.concat((dftext, dftext_i))  if dftext is not None else dftext_i
        save_features(dftext_i, 'dftext_' + coltext_i, path_features_store)

    log(dftext.head(6))
    save_features(dftext, 'dftext', path_features_store)
"""



