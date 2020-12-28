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


def pd_income_processor(df, col, pars):
  """
     Processor for DSA@

  """
  df = df.rename(columns = {'39':'age' , ' State-gov':'workclass', ' 77516':'final_weight', ' Bachelors':'education', ' 13':'education-num', ' Never-married':'marital-status', ' Adm-clerical':'occupation', ' Not-in-family':'relationship', ' White':'race', ' Male':'sex', ' 2174':'capital-gain', ' 0':'capital-loss', ' 40':'hours-per-week', ' United-States':'native-country', ' <=50K':'status'})


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



df         = pd.read_csv(folder+'train_csv.csv', delimiter=',')
df         = pd_income_processor(df, list(df.columns), pars={} )
feature_tr = df.drop(["status"],axis=1)
target_tr  = df[["status","id"]]
feature_tr.to_csv( "train/features.csv", index=False)
target_tr.to_csv(  "train/target.csv",index=False)




df_test     = pd.read_csv(folder+'test_csv.csv', delimiter=',')
df_test = pd_income_processor(df_test, list(df_test.columns), pars={} )
feature_test=df_test.drop(["status"],axis=1)
target_test=df_test[["status","id"]]
feature_test.to_csv( "test/features.csv", index=False)
target_test.to_csv(  "test/target.csv",index=False)







   

def pd_income_processor(df, col, pars):
  """


  """
  df = df.rename(columns = {'39':'age' , ' State-gov':'workclass', ' 77516':'final_weight', ' Bachelors':'education', ' 13':'education-num', ' Never-married':'marital-status', ' Adm-clerical':'occupation', ' Not-in-family':'relationship', ' White':'race', ' Male':'sex', ' 2174':'capital-gain', ' 0':'capital-loss', ' 40':'hours-per-week', ' United-States':'native-country', ' <=50K':'status'})
  df_test = df_test.rename(columns = {'25':'age' , ' Private':'workclass', ' 226802':'final_weight', ' 11th':'education', ' 7':'education-num', ' Never-married':'marital-status', ' Machine-op-inspct':'occupation', ' Own-child':'relationship', ' Black':'race', ' Male':'sex', ' 0':'capital-gain', ' 0.1':'capital-loss', ' 40':'hours-per-week', ' United-States':'native-country', ' <=50K.':'status'})


  df.drop(['education'], axis=1, inplace = True)
  df_test.drop(['education'], axis=1, inplace = True)

  # The replacing of "?" with "nan" on the dataset
  df.replace(" ?",np.NaN,inplace=True)
  df_test.replace(" ?",np.NaN,inplace=True)

  # Converting the value of the target column to a 0-1
  df.status = [ 0 if each == " <=50K" else 1 for each in df.status]
  df_test.status = [ 0 if each == " <=50K" else 1 for each in df_test.status]


  df["id"]=[i+1 for i in range(len(df))]
  df_test["id"]=[i+1 for i in range(len(df_test))]


  num_col=['age', 'final_weight', 'capital-gain', 'capital-loss', 'hours-per-week']
  cat_col=['occupation','workclass','native-country','education-num','marital-status','relationship','race','sex','status']
  sparse_col=['capital-gain', 'capital-loss']

  # Find IQR and implement to numericals and sparse columns seperately
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
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

      df_test[col] = np.where(df_test[col] > upper_bound, 1.25*upper_bound, df_test[col])
      df_test[col] = np.where(df_test[col] < lower_bound, 0.75*lower_bound, df_test[col])

    return dfnew, col_pars





