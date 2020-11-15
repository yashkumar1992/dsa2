"""%load_ext autoreload
%autoreload
%matplotlib inline
%config IPCompleter.greedy=True"""
import warnings,copy
from source import run_train
from source import run_inference
from source import config_model
from source.models import model_sklearn
warnings.filterwarnings('ignore')



############################################################################
from allimport import *
from source import util_feature


############################################################################
print( os.getcwd())
root      = os.getcwd()
dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


############################################################################
############################################################################
run_train.run_train('titanic_lightgbm', '/input/titanic/train/', '/data/output/a01_lightgbm_classifier/', 
                	path_config_model="source/config_model.py" , n_sample = -1)


from source.util_feature import load
from source.models import model_sklearn as modelx
import sys
from source import models
sys.modules['models']=models
model_tag =  "a01_lightgbm_classifier"




############################################################################
############################################################################
#### Load model
dir_model    = dir_data + f"/output/{model_tag}/"
modelx.model = load( dir_model + "/model/model.pkl" )
stats        = load( dir_model + "/model/info.pkl" )
colsX        = load( dir_model + "/model/colsX.pkl"   )
coly         = load( dir_model + "/model/coly.pkl"   )
print(stats)
print(modelx.model.model)

### Metrics on test data
stats['metrics_test']

#### Loading training data  ###############################################
dfX     = pd.read_csv(dir_model + "/check/dfX.csv")
dfy     = dfX[coly]
colused = colsX

dfXtest = pd.read_csv(dir_model + "/check/dfXtest.csv")
dfytest = dfXtest[coly]

print(dfX.shape,  dfXtest.shape )


#### Feature importance on training data
lgb_featimpt_train,_ = util_feature.feature_importance_perm(modelx, dfX[colused], dfy, colused, n_repeats=1,  
	                                                        scoring='accuracy_score' )

print(lgb_featimpt_train)

#! python source/run_inference.py  run_predict  --model_name  LGBMRegressor  --n_sample 1000   --path_model /data/output/a01_lightgbm_huber/    --path_output /data/output/pred_a01_lightgbm_huber/    --path_data /data/input/train/

run_inference.run_predict('LGBMClassifier', '/data/output/a01_lightgbm_classifier/','/input/titanic/test/', '/data/output/a01_lightgbm_classifier/', -1)





