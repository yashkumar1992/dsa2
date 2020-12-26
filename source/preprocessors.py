# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""


"""
import warnings
warnings.filterwarnings('ignore')
import sys, gc, os, pandas as pd, json, copy

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


#### Debuging state (Ture/False)
DEBUG_=True

####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)

def logs(*s):
    if DEBUG_:
        print(*s, flush=True)


def log_pd(df, *s, n=0, m=1):
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump,  df.head(n), flush=True)


from util_feature import  save, load_function_uri, load
import util_feature
####################################################################################################
####################################################################################################
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
    list1 = col
    # list1 = []
    # list1.append(col)
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
    logs('----col-----\n', col)
    coltext_freq = df[col].apply(str).apply(lambda x: pd.value_counts(x.split(sep))).sum(axis=0).reset_index()
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
    """
    df : Datframe
    col : list of columns
    pars : dict of pars

    """
    from utils import util_text, util_model

    #### Load pars ###################################################################
    path_pipeline        = pars.get('path_pipeline', None)
    word_tokeep_dict_all = load(  path_pipeline + "/word_tokeep_dict_all.pkl" )  if path_pipeline is not None else {}
    # dftext_tdidf_all = load(f'{path_pipeline}/dftext_tdidf.pkl') if  path_pipeline else None
    # dftext_svd_list_all      = load(f'{path_pipeline}/dftext_svd.pkl')   if  path_pipeline else None
    dimpca       = pars.get('dimpca', 2)
    word_minfreq = pars.get('word_minfreq', 3)

    #### Process  ####################################################################
    stopwords           = nlp_get_stopwords()
    dftext              = pd_coltext_clean(df, col, stopwords= stopwords , pars=pars)
    dftext_svd_list_all = None
    dftext_tdidf_all    = None

    ### Processing each of text columns to create a bag of word/to load the bag of word -> tf-idf -> svd
    for col_ in col:

            if path_pipeline is not None:
                ### If it is in Inference step, use the saved bag of word for the column `col_`
                word_tokeep = word_tokeep_dict_all[col_]

            else:
                ### If it is not, create a bag of word
                coltext_freq, word_tokeep = pd_coltext_wordfreq(df, col_, stopwords, ntoken=100)  ## nb of words to keep
                word_tokeep_dict_all[col_] = word_tokeep  ## save the bag of wrod for `col_` in a dict

            dftext_tdidf_dict, word_tokeep_dict = util_text.pd_coltext_tdidf(dftext, coltext=col_, word_minfreq= word_minfreq,
                                                                             word_tokeep = word_tokeep,
                                                                             return_val  = "dataframe,param")

            dftext_tdidf_all = pd.DataFrame(dftext_tdidf_dict) if dftext_tdidf_all is None else pd.concat((dftext_tdidf_all,pd.DataFrame(dftext_tdidf_dict)),axis=1)
            log(word_tokeep_dict)

            ###  Dimesnion reduction for Sparse Matrix
            dftext_svd_list, svd_list = util_model.pd_dim_reduction(dftext_tdidf_dict,
                                                           colname        = None,
                                                           model_pretrain = None,
                                                           colprefix      = col_ + "_svd",
                                                           method         = "svd",  dimpca=dimpca,  return_val="dataframe,param")

            dftext_svd_list_all = dftext_svd_list if dftext_svd_list_all is None else pd.concat((dftext_svd_list_all,dftext_svd_list),axis=1)
    #################################################################################

    ###### Save and Export ##########################################################
    if 'path_features_store' in pars:
            save_features(dftext_svd_list_all, 'dftext_svd' + "-" + str(col), pars['path_features_store'])
            # save(dftext_svd_list_all,  pars['path_pipeline_export'] + "/dftext_svd.pkl")
            # save(dftext_tdidf_all,     pars['path_pipeline_export'] + "/dftext_tdidf.pkl" )
            save(word_tokeep_dict_all,     pars['path_pipeline_export'] + "/word_tokeep_dict_all.pkl" )

    col_pars = {}
    col_pars['cols_new'] = {
     # 'coltext_tdidf'    : dftext_tdidf_all.columns.tolist(),       ### list
     'coltext_svd'      : dftext_svd_list_all.columns.tolist()      ### list
    }

    dftext_svd_list_all.index = dftext.index
    # return pd.concat((dftext_svd_list_all,dftext_svd_list_all),axis=1), col_pars
    return dftext_svd_list_all, col_pars



##### Filtering / cleaning rows :   #########################################################
def pd_filter_rows(df, col, pars):
    import re
    coly = col
    filter_pars =  pars
    def isfloat(x):
        x = re.sub("[!@,#$+%*:()'-]", "", x)
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
    df[coly] = df[coly].astype('float32')
    del df['_isfloat']
    logs("----------df[coly]------------",df[coly])
    ymin, ymax = pars.get('ymin', -9999999999.0), pars.get('ymax', 999999999.0)
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]

    ##### Label processing   ####################################################################
    y_norm_fun = None
    # Target coly processing, Normalization process  , customize by model
    log("y_norm_fun preprocess_pars")
    y_norm_fun = pars.get('y_norm_fun', None)
    if y_norm_fun is not None:
        df[coly] = df[coly].apply(lambda x: y_norm_fun(x))
        # save(y_norm_fun, f'{path_pipeline_export}/y_norm.pkl' )

    if pars.get('path_features_store', None) is not None:
        path_features_store = pars['path_features_store']
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

    pars = { 'pipe_list': [ {'name': 'fillna', 'naval' : 0.0 }, {'name': 'minmax'} ]}
    dfnum_norm, colnum_norm = pd_colnum_normalize(df, colname=colnum,  pars=pars, suffix = "_norm",
                                                  return_val="dataframe,param")
    log(colnum_norm)
    if pars.get('path_features_store', None) is not None:
        path_features_store = pars['path_features_store']
        save_features(dfnum_norm, 'dfnum_norm', path_features_store)
    return dfnum_norm, colnum_norm


def pd_colnum_bin(df, col, pars):
    from util_feature import  pd_colnum_tocat

    path_pipeline = pars.get('path_pipeline', False)
    colnum_binmap  = load(f'{path_pipeline}/colnum_binmap.pkl') if  path_pipeline else None
    log(colnum_binmap)

    colnum = col

    log("### colnum Map numerics to Category bin  ###########################################")
    dfnum_bin, colnum_binmap = pd_colnum_tocat(df, colname=colnum, colexclude=None, colbinmap=colnum_binmap,
                                               bins=10, suffix="_bin", method="uniform",
                                               return_val="dataframe,param")
    log(colnum_binmap)
    ### Renaming colunm_bin with suffix
    colnum_bin = [x + "_bin" for x in list(colnum_binmap.keys())]
    log(colnum_bin)

    if 'path_features_store' in pars:
        scol = "_".join(col[:5])
        save_features(dfnum_bin, 'colnum_bin' + "-" + scol, pars['path_features_store'])
        save(colnum_binmap,  pars['path_pipeline_export'] + "/colnum_binmap.pkl" )
        save(colnum_bin,     pars['path_pipeline_export'] + "/colnum_bin.pkl" )


    col_pars = {}
    col_pars['colnumbin_map'] = colnum_binmap
    col_pars['cols_new'] = {
     'colnum'     :  col ,    ###list
     'colnum_bin' :  colnum_bin       ### list
    }
    return dfnum_bin, col_pars


def pd_colnum_binto_onehot(df, col=None, pars=None):
    assert isinstance(col, list) and isinstance(df, pd.DataFrame)

    dfnum_bin = df[col]
    colnum_bin = col

    path_pipeline = pars.get('path_pipeline', False)
    colnum_onehot = load(f'{path_pipeline}/colnum_onehot.pkl') if  path_pipeline else None


    log("###### colnum bin to One Hot  #################################################")
    from util_feature import  pd_col_to_onehot
    dfnum_hot, colnum_onehot = pd_col_to_onehot(dfnum_bin[colnum_bin], colname=colnum_bin,
                                                colonehot=colnum_onehot, return_val="dataframe,param")
    log(colnum_onehot)

    if 'path_features_store' in pars :
        save_features(dfnum_hot, 'colnum_onehot', pars['path_features_store'])
        save(colnum_onehot,  pars['path_pipeline_export'] + "/colnum_onehot.pkl" )

    col_pars = {}
    col_pars['colnum_onehot'] = colnum_onehot
    col_pars['cols_new'] = {
     # 'colnum'        :  col ,    ###list
     'colnum_onehot' :  colnum_onehot       ### list
    }
    return dfnum_hot, col_pars



def pd_colcat_to_onehot(df, col=None, pars=None):
    dfbum_bin = df[col]
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

    path_pipeline = pars.get('path_pipeline', False)
    colcat_onehot = load(f'{path_pipeline}/colcat_onehot.pkl') if  path_pipeline else None

    colcat = col
    log("#### colcat to onehot")
    dfcat_hot, colcat_onehot = util_feature.pd_col_to_onehot(df[colcat], colname=colcat,
                                                colonehot=colcat_onehot, return_val="dataframe,param")
    log(dfcat_hot[colcat_onehot].head(5))

    if 'path_features_store' in pars :
        path_features_store = pars['path_features_store']
        save_features(dfcat_hot, 'colcat_onehot', path_features_store)
        save(colcat_onehot,  pars['path_pipeline_export'] + "/colcat_onehot.pkl" )
        save(colcat,         pars['path_pipeline_export'] + "/colcat.pkl" )

    col_pars = {}
    col_pars['colcat_onehot'] = colcat_onehot
    col_pars['cols_new'] = {
     # 'colnum'        :  col ,    ###list
     'colcat_onehot' :  colcat_onehot       ### list
    }

    print("ok ------------")
    return dfcat_hot, col_pars



def pd_colcat_bin(df, col=None, pars=None):
    # dfbum_bin = df[col]
    path_pipeline = pars.get('path_pipeline', False)
    colcat_bin_map = load(f'{path_pipeline}/colcat_bin_map.pkl') if  path_pipeline else None

    colcat = col
    log("#### Colcat to integer encoding ")
    dfcat_bin, colcat_bin_map = util_feature.pd_colcat_toint(df[colcat], colname=colcat,
                                                            colcat_map=  colcat_bin_map ,
                                                            suffix="_int")
    colcat_bin = list(dfcat_bin.columns)
    ##### Colcat processing   ################################################################
    colcat_map = util_feature.pd_colcat_mapping(df, colcat)
    log(df[colcat].dtypes, colcat_map)


    if 'path_features_store' in pars :
       save_features(dfcat_bin, 'dfcat_bin', pars['path_features_store'])
       save(colcat_bin_map,  pars['path_pipeline_export'] + "/colcat_bin_map.pkl" )
       save(colcat_bin,      pars['path_pipeline_export'] + "/colcat_bin.pkl" )


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

    dfcat_hot = pars['dfcat_hot']
    dfnum_hot = pars['dfnum_hot']
    colid     = pars['colid']

    try :
       df_onehot = dfcat_hot.join(dfnum_hot, on=colid, how='left')
    except :
       df_onehot = copy.deepcopy(dfcat_hot)

    colcross_single = pars['colcross_single']
    if  'path_pipeline' in pars :   #### Load existing column list
       colcross_single = load( pars['path_pipeline']  +f'/colcross_single_onehot_select.pkl')

    colcross_single_onehot_select = []
    for t in list(df_onehot.columns):
       for c1 in colcross_single:
           if c1 in t:
               colcross_single_onehot_select.append(t)


    df_onehot = df_onehot[colcross_single_onehot_select ]
    dfcross_hot, colcross_pair = pd_feature_generate_cross(df_onehot, colcross_single_onehot_select,
                                                           pct_threshold=0.02,  m_combination=2)
    log(dfcross_hot.head(2).T)
    colcross_pair_onehot = list(dfcross_hot.columns)

    if 'path_features_store' in pars:
        save_features(dfcross_hot, 'colcross_onehot', pars['path_features_store'])
        save(colcross_single_onehot_select, pars['path_pipeline_export'] + '/colcross_single_onehot_select.pkl')
        save(colcross_pair,                 pars['path_pipeline_export'] + '/colcross_pair.pkl')

    col_pars = {}
    col_pars['colcross_pair'] = colcross_pair
    col_pars['cols_new'] = {
     # 'colcross_single'     :  col ,    ###list
     'colcross_pair' :  colcross_pair       ### list
    }
    return dfcross_hot, col_pars



def pd_coldate(df, col, pars):
    log("##### Coldate processing   #############################################################")
    from utils import util_date
    coldate = col
    dfdate  = None
    for coldate_i in coldate :
        dfdate_i = util_date.pd_datestring_split( df[[coldate_i]] , coldate_i, fmt="auto", return_val= "split" )
        dfdate   = pd.concat((dfdate, dfdate_i),axis=1)  if dfdate is not None else dfdate_i
        # if 'path_features_store' in pars :
        #    path_features_store = pars['path_features_store']
        #    #save_features(dfdate_i, 'dfdate_' + coldate_i, path_features_store)

    if 'path_features_store' in pars :
        save_features(dfdate, 'dfdate', pars['path_features_store'])

    col_pars = {}
    col_pars['cols_new'] = {
        # 'colcross_single'     :  col ,    ###list
        'dfdate': list(dfdate.columns)  ### list
    }
    return dfdate, col_pars


def pd_colcat_minhash(df, col, pars):
    """
       MinHash Algo for category
       https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087

    """
    colcat              = col

    pars_minhash = {'n_component' : [4, 2], 'model_pretrain_dict' : None,}
    if 'path_pipeline_export' in pars :
        try :
            pars_minhash = load( pars['path_pipeline_export'] + '/colcat_minhash_pars.pkl')
        except : pass

    log("#### Colcat to Hash encoding #############################################")
    from utils import util_text
    dfcat_bin, col_hash_model= util_text.pd_coltext_minhash(df[colcat], colcat,
                                                            return_val="dataframe,param", **pars_minhash )
    colcat_minhash = list(dfcat_bin.columns)
    log(col_hash_model)
    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(dfcat_bin, 'dfcat_minhash', pars['path_features_store'])
       save(col_hash_model, pars['path_pipeline_export'] + "/colcat_minhash_model.pkl" )
       save(colcat_minhash, pars['path_pipeline_export'] + "/colcat_minhash.pkl" )
       save(pars_minhash,   pars['path_pipeline_export'] + "/colcat_minhash_pars.pkl" )


    col_pars = {}
    col_pars['col_hash_model'] = col_hash_model
    col_pars['cols_new'] = {
     'colcat_minhash' :  colcat_minhash  ### list
    }
    return dfcat_bin, col_pars




def pd_coltext_universal_google(df, col, pars={}):
    """
     # Universal sentence encoding from Tensorflow
       Text ---> Vectors
    from source.preprocessors import  pd_coltext_universal_google
    https://tfhub.dev/google/universal-sentence-encoder-multilingual/3

    #@title Setup Environment
    #latest Tensorflow that supports sentencepiece is 1.13.1
    !pip uninstall --quiet --yes tensorflow
    !pip install --quiet tensorflow-gpu==1.13.1
    !pip install --quiet tensorflow-hub
    pip install --quiet tf-sentencepiece, simpleneighbors
    !pip install --quiet simpleneighbors

    # df : dataframe
    # col : list of text colnum names
    pars
    """
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text
    #from tqdm import tqdm #progress bar

    uri_default = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/1"
    uri         = pars.get("url_model", uri_default )
    use    = hub.load( uri )
    dfall  = None
    for coli in col[:1] :
        X = []
        for r in (df[coli]):
            if pd.isnull(r)==True :
                r=""
            emb = use(r)
            review_emb = tf.reshape(emb, [-1]).numpy()
            X.append(review_emb)

        dfi   = pd.DataFrame(X, columns= [ coli + "_" + str(i) for i in range(X.shape[1])   ])
        dfall = pd.concat((dfall, dfi))  if dfall is not None else dfi

    coltext_embed = list(dfall.columns)
    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(dfall, 'dftext_embed', pars['path_features_store'])
       save(coltext_embed, pars['path_pipeline_export'] + "/coltext_universal_google.pkl" )

    col_pars = {'model_encoder' : uri}
    col_pars['cols_new']      = {
     'coltext_universal_google' :  coltext_embed ### list
    }
    return dfall, col_pars





if __name__ == "__main__":
    import fire
    fire.Fire()


