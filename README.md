### Install

     pip install pyro-ppl lightgbm pandas scikit-learn scipy matplotlib




### Basic usage
    cd dsa2
    python run.py data_profile --config_uri titanic_classifier.py::titanic_lightgbm   > zlog/log-titanic.txt 2>&1
    python run.py preprocess   --config_uri titanic_classifier.py::titanic_lightgbm   > zlog/log-titanic.txt 2>&1
    python run.py train        --config_uri titanic_classifier.py::titanic_lightgbm   > zlog/log-titanic.txt 2>&1
    python run.py predict      --config_uri titanic_classifier.py::titanic_lightgbm   > zlog/log-titanic.txt 2>&1



### Basic usage 2
    python  titanic_classifier.py  data_profile
    python  titanic_classifier.py  preprocess
    python  titanic_classifier.py  train
    python  titanic_classifier.py  check
    python  titanic_classifier.py  predict
    python  titanic_classifier.py  run_all


### data/input  : Input data format

    data/input/titanic/raw/  : the raw files
    data/input/titanic/raw2/ : the raw files  split manually


    data/input/titanic/train/ :   features.zip ,  target.zip, cols_group.json  names are FIXED
             features.zip or features.parquet  :  csv file of the inputs
             target.zip   or target.parquet    :  csv file of the label to predict.


    data/input/titanic/test/ :   
             features.zip or parquet format  , used for predictions

    File names Are FIXED, please create sub-folder  


###  Model, train, inference :
   All are defined in a single model_dictionnary containing all




###  Column Group for model preprocessing / training/inference :

    *Titanic dataframe structure (example:
                 Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
    PassengerId                                                                                                                                           
    1                   0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
    2                   1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
    3                   1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
    4                   1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
    5                   0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S


    (1) Initial Manual Column Mapping  :   'cols_input_type' 
     From Raw data --> "colid","colnum","colcat","coldate","coltext","coly","colcross"
       
       |-"colid"    --> index or id of each row (e.g. ["PassengerId"])

       |-"coly"     --> target column or y column (e.g. ["Survived"])


       |-"colnum"   --> columns with float or interger numbers (e.g. ["Pclass", "Age", "SibSp", "Parch", "Fare"])
       |-"colcat"   --> columns with string labels (e.g. ["Sex", "Embarked"])

       |-"coldate"  --> columns with date format data
       |-"coltext"  --> columns with text data (e.g. ["Ticket", "Name"])


       |-"colcross" --> columns to be checked for feature crosses
                        (e.g. ["Name", "Sex", "Ticket", "Embarked", "Pclass", "Age", "SibSp", "Parch", "Fare"])
    

     
    (2) Columns feature family for model training  :   "cols_model_group"
        "cols_model_group" --> column family used for Model Training 

      "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns                        
      "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns                        
      'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns            
      'coldate',
      'coltext',            



###  Preprocessing pipeline dataframe ( in source/run_preprocess.py)   :


    *Preprocessing as follow    in  source/run_preprocess.py
         Raw Columns   --->  Feature columns family

        "colnum"    --> "colnum_bin" --> "colnum_onehot"
            |--------------------------> "colnum_onehot"
            
        "colcat"    --> "colcat_bin" --> "colcat_onehot"
            |--------------------------> "colcat_onehot"

     Default pipeline options are considered in 

     pipe_default = [ 'filter','label','dfnum_bin', 'dfnum_hot', 'dfcat_bin', 'dfcat_hot', 'dfcross_hot'] :



    'filter':     Input:  ymin and ymax values  and does filtering of dataset (coly).
    'label' :     Input:  y_norm_fun value from model dictionary
                   applies normalization function on coly


    'dfnum_bin' : Input:  a dataframe with selected numerical columns, creates categorical bins, returns dataframe with new columns (dfnum_bin)
    'dfnum_hot' : Input:  a dataframe dfnum_bin, returns one hot matrix as dataframe dfnum_hot


    'dfcat_bin' : Input:  a dataframe with categorical columns, returns dataframe dfcat_bin with numerical values
    'dfcat_hot' : Input:  a dataframe with categorical columns, returns one hot matrix as dataframe dfcat_hot


    'dfcross_hot' : Input:  a data frame of numerical and categorical one hot encoded columns with defined cross columns, returns dataframe df_cross_hot

    

### Command line usage advanced
    cd dsa2
    source activate py36 
    python source/run_train.py  run_train   --n_sample 100  --model_name lightgbm  --path_config_model source/config_model.py  --path_output /data/output/a01_test/     --path_data /data/input/train/    


    source activate py36 
    python source/run_inference.py  run_predict  --n_sample 1000  --model_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data /data/input/train/








### source/  : code source CLI to train/predict.
```
   run_feature_profile.py : CLI Pandas profiling
   run_preprocess.py      : CLI for feature preprocessing
   run_train.py :           CLI to train any model, any data (model  data agnostic )
   run_inference.py :       CLI to predict with any model, any data (model  data agnostic )




```



### source/models/  : Generic API to access models.
```
   One file python file per model.

   models/model_sklearn.py      :   generic module as class, which wraps any sklearn API type model.
   models/model_bayesian_pyro.py :  generic model as class, which wraps Bayesian regression in Pyro/Pytorch.

   Method of the moddule/class
       .init
       .fit()
       .predict()


```




