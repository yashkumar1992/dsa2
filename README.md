## Install


pip install pyro-ppl lightgbm pandas scikit-learn scipy matplotlib



## Command line

cd dsa2
! source activate py36 
python source/run_train.py  run_train   --n_sample 100  --model_name lightgbm  --path_config_model source/config_model.py  --path_output /data/output/a01_test/     --path_data /data/input/train/    


! activate py36 
python source/run_inference.py  run_predict  --n_sample 1000  --model_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data /data/input/train/



### data/input  : Input data
   cols_group.json : name of columns per feature category.
   features.zip : csv file of the inputs
   target-values : csv file of the label to predict.



## source/  : code source CLI to train/predict with the models.
```
   run_feature_profile.py : CLI Pandas profiling
   run_train.py :         CLI to train any model, any data (model  data agnostic )
   run_inference.py :     CLI to predict with any model, any data (model  data agnostic )


   config_model.py   :  file containing custom parameter for each specific model.


```



## source/models/  : Generic API to access models.
```
   One file python file per model.

   model_sklearn.py      :  generic module as class, which wraps any sklearn API type model.
   model_bayesian_pyro.py :  generic model as class, which wraps Bayesian regression in Pyro/Pytorch.

   Method of the moddule/class
       .init
       .fit()
       .predict()


```




