import pandas as pd
import random, os, sys
import numpy as np



#### Add path for python import  #######################################
path_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
print("path_repo_root", path_repo_root)
sys.path.append( path_repo_root)
from util_feature import  save, load_function_uri, load 		
########################################################################


folder     = 'raw/'
df         = pd.read_csv(folder+'train_csv.csv', delimiter=',')
df_test     = pd.read_csv(folder+'test_csv.csv', delimiter=',')

"""
	Rename columns for train and test 
"""
df = df.rename(columns = {'39':'age' , ' State-gov':'workclass', ' 77516':'final_weight', ' Bachelors':'education', ' 13':'education-num', ' Never-married':'marital-status', ' Adm-clerical':'occupation', ' Not-in-family':'relationship', ' White':'race', ' Male':'sex', ' 2174':'capital-gain', ' 0':'capital-loss', ' 40':'hours-per-week', ' United-States':'native-country', ' <=50K':'status'})
df_test = df_test.rename(columns = {'25':'age' , ' Private':'workclass', ' 226802':'final_weight', ' 11th':'education', ' 7':'education-num', ' Never-married':'marital-status', ' Machine-op-inspct':'occupation', ' Own-child':'relationship', ' Black':'race', ' Male':'sex', ' 0':'capital-gain', ' 0.1':'capital-loss', ' 40':'hours-per-week', ' United-States':'native-country', ' <=50K.':'status'})



##########################################################################################
##########################################################################################
def pd_cleanup(df, col, pars):
  df.drop(['education'], axis=1, inplace = True)
  df.replace(" ?",np.NaN,inplace=True)

  # Converting the value of the target column to a 0-1
  df.status = [ 0 if each == " <=50K" else 1 for each in df.status]
  df["id"]=[i+1 for i in range(len(df))]
  return df




def pd_normalize_quantile(df, col=['age', 'final_weight', 'capital-gain', 'capital-loss', 'hours-per-week'], pars={}):
  """
     Processor for DSA@
  """
  df = df[col]

  num_col = col
  # num_col=['age', 'final_weight', 'capital-gain', 'capital-loss', 'hours-per-week']
  sparse_col= pars.get('colsparse', ['capital-gain', 'capital-loss'] )

  # Find IQR and implement to numericals and sparse columns seperately
  Q1  = df.quantile(0.25)
  Q3  = df.quantile(0.75)
  IQR = Q3 - Q1

  for col in num_col:
    if col in sparse_col:
      nonsparse_data = pd.DataFrame(df[df[col] !=df[col].mode()[0]][col])
      if nonsparse_data[col].quantile(0.25) < df[col].mode()[0]: #Unexpected case
        lower_bound_sparse = nonsparse_data[col].quantile(0.25)
      else:
        lower_bound_sparse = df[col].mode()[0]

      if nonsparse_data[col].quantile(0.75) < df[col].mode()[0]: #Unexpected case
        upper_bound_sparse = df[col].mode()[0]
      else:
        upper_bound_sparse = nonsparse_data[col].quantile(0.75)

      number_of_outliers = len(df[(df[col] < lower_bound_sparse) | (df[col] > upper_bound_sparse)][col])

      if number_of_outliers > 0:

        df.loc[df[col] < lower_bound_sparse,col] = lower_bound_sparse*0.75 #--> MAIN DF CHANGED

        df.loc[df[col] > upper_bound_sparse,col] = upper_bound_sparse*1.25 # --> MAIN DF CHANGED
    else:
      lower_bound = df[col].quantile(0.25) - 1.5*IQR[col]
      upper_bound = df[col].quantile(0.75) + 1.5*IQR[col]

      df[col] = np.where(df[col] > upper_bound, 1.25*upper_bound, df[col])
      df[col] = np.where(df[col] < lower_bound, 0.75*lower_bound, df[col])


  colnew   = [ t + "_norm" for t in df.columns ]
  pars_new = {'lower_bound' : lower_bound, 'upper_bound': upper_bound}
  dfnew    = df

  encoder_model = None
  ###################################################################################
  if 'path_features_store' in pars and 'path_pipeline_export' in pars:
     # pass
     #save_features(df, 'dfcat_encoder', pars['path_features_store'])
     # save(colnew,    pars['path_pipeline_export']  + "/colnum_encoder_model.pkl" )
     save(pars_new,  pars['path_pipeline_export']   + "/colnum_quantile_norm_new.pkl" )


  col_pars = {}
  col_pars['model']    = encoder_model
  col_pars['pars']     = pars_new
  col_pars['cols_new'] = {
   'colnum_normalize_quantile' :  colnew  ### list
  }

  #dfnew    = df.drop(["status"],axis=1)
  return dfnew, col_pars












"""
def pd_income_processor(df, col, pars):
  # Processor for DSA@
  #df = df.rename(columns = {'39':'age' , ' State-gov':'workclass', ' 77516':'final_weight', ' Bachelors':'education', ' 13':'education-num', ' Never-married':'marital-status', ' Adm-clerical':'occupation', ' Not-in-family':'relationship', ' White':'race', ' Male':'sex', ' 2174':'capital-gain', ' 0':'capital-loss', ' 40':'hours-per-week', ' United-States':'native-country', ' <=50K':'status'})


  df.drop(['education'], axis=1, inplace = True)

  # The replacing of "?" with "nan" on the dataset
  df.replace(" ?",np.NaN,inplace=True)

  # Converting the value of the target column to a 0-1
  df.status = [ 0 if each == " <=50K" else 1 for each in df.status]


  df["id"]=[i+1 for i in range(len(df))]


  num_col=['age', 'final_weight', 'capital-gain', 'capital-loss', 'hours-per-week']
  cat_col=['occupation','workclass','native-country','education-num','marital-status','relationship','race','sex','status']
  sparse_col=['capital-gain', 'capital-loss']

  # Find IQR and implement to numericals and sparse columns seperately
  Q1  = df.quantile(0.25)
  Q3  = df.quantile(0.75)
  IQR = Q3 - Q1

  for col in num_col:
    if col in sparse_col:
      nonsparse_data = pd.DataFrame(df[df[col] !=df[col].mode()[0]][col])
      if nonsparse_data[col].quantile(0.25) < df[col].mode()[0]: #Unexpected case
        lower_bound_sparse = nonsparse_data[col].quantile(0.25)
      else:
        lower_bound_sparse = df[col].mode()[0]

      if nonsparse_data[col].quantile(0.75) < df[col].mode()[0]: #Unexpected case
        upper_bound_sparse = df[col].mode()[0]
      else:
        upper_bound_sparse = nonsparse_data[col].quantile(0.75)

      number_of_outliers = len(df[(df[col] < lower_bound_sparse) | (df[col] > upper_bound_sparse)][col])

      if number_of_outliers > 0:

        df.loc[df[col] < lower_bound_sparse,col] = lower_bound_sparse*0.75 #--> MAIN DF CHANGED

        df.loc[df[col] > upper_bound_sparse,col] = upper_bound_sparse*1.25 # --> MAIN DF CHANGED
    else:
      lower_bound = df[col].quantile(0.25) - 1.5*IQR[col]
      upper_bound = df[col].quantile(0.75) + 1.5*IQR[col]

      df[col] = np.where(df[col] > upper_bound, 1.25*upper_bound, df[col])
      df[col] = np.where(df[col] < lower_bound, 0.75*lower_bound, df[col])


    pars_new = {}
    dfnew    = df

    colX = [  t for t in  dfnew.columns if t not in [ 'status', 'id' ]]
    coly = [  t for t in  dfnew.columns if t  in [ 'status'  ]]
    colXy_income = [  t for t in  dfnew.columns if t  in [ 'id'  ]]

    encoder_model = None
    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       pass
       #save_features(df, 'dfcat_encoder', pars['path_features_store'])
       #save(encoder,       pars['path_pipeline_export']   + "/colcat_encoder_model.pkl" )
       #save(colcat_encoder,  pars['path_pipeline_export'] + "/colcat_encoder.pkl" )

 
    col_pars = {}
    col_pars['model'] = encoder_model
    col_pars['cols_new'] = {
     'colXy_income' :  colXy_income  ### list
    }

    #dfnew    = df.drop(["status"],axis=1)
    return dfnew, col_pars
"""


"""
	Saving files to csv and also zip format
"""

df = pd_cleanup(df, col=None, pars=None)

df = pd_normalize_quantile(df,
                           col=['age', 'final_weight', 'capital-gain', 'capital-loss', 'hours-per-week'], pars={})


#df         = pd_income_processor(df, list(df.columns), pars={} )
feature_tr = df[0].drop(["status"],axis=1)
target_tr  = df[0][["status","id"]]
feature_tr.to_csv( "train/features.csv", index=False)
target_tr.to_csv(  "train/target.csv",index=False)


features = dict(method='zip',archive_name='features.csv')  
target = dict(method='zip',archive_name='target.csv')
feature_tr.to_csv('train/features.zip', index=False, compression=features) 
target_tr.to_csv('train/target.zip', index=False,compression=target)




#####
df_test = pd_cleanup(df_test, col=None, pars=None)

df_test = pd_normalize_quantile(df_test,
                  col=['age', 'final_weight', 'capital-gain', 'capital-loss', 'hours-per-week'], pars={})

# df_test      = pd_income_processor(df_test, list(df_test.columns), pars={} )
feature_test = df_test[0].drop(["status"],axis=1)
target_test  = df_test[0][["status","id"]]
feature_test.to_csv( "test/features.csv", index=False)
target_test.to_csv(  "test/target.csv",index=False)

feature_test.to_csv('test/features.zip', index=False,compression=features) 
target_test.to_csv('test/target.zip', index=False, compression=target)













