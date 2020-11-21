### Install

     pip install pyro-ppl lightgbm pandas scikit-learn scipy matplotlib




### Basic usage
    cd dsa2
    python run.py data_profile --config_uri titanic_classifier.py::titanic_lightgbm
    python run.py preprocess   --config_uri titanic_classifier.py::titanic_lightgbm
    python run.py train        --config_uri titanic_classifier.py::titanic_lightgbm
    python run.py predict      --config_uri titanic_classifier.py::titanic_lightgbm



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


###  Column Group for model  :
    ['colid',
    "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
    
    
    "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
    
    
    'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns

    'coldate',
    'coltext',

    "coly"
    ]


###  Column Preprocessing pipeline dataframe   :
     pipe_list = [ 'dfnum_bin', 'dfnum_hot', 'dfcat_bin', 'dfcat_hot', 'dfcross_hot',
                   'dfdate',  'dftext'  ] :



### Command line usage advanced
    cd dsa2
    source activate py36 
    python source/run_train.py  run_train   --n_sample 100  --model_name lightgbm  --path_config_model source/config_model.py  --path_output /data/output/a01_test/     --path_data /data/input/train/    


    source activate py36 
    python source/run_inference.py  run_predict  --n_sample 1000  --model_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data /data/input/train/








### source/  : code source CLI to train/predict.
```
   run_feature_profile.py : CLI Pandas profiling
   run_train.py :           CLI to train any model, any data (model  data agnostic )
   run_inference.py :       CLI to predict with any model, any data (model  data agnostic )


   config_model.py   :  file containing custom parameter for each specific model.
                        Please insert your model config there :
                           titanic_lightgbm


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




