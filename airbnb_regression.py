# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to titatinic dataet
All in one file config
!  python airbnb_regression.py  train
!  python airbnb_regression.py  check
!  python airbnb_regression.py  predict
"""
import warnings, copy
warnings.filterwarnings('ignore')
import os, sys
import pandas as pd

############################################################################
from source import util_feature
import copy

###### Path ################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


####################################################################################
config_file  = "airbnb_regression.py"
data_name    = "airbnb"


config_name  = 'airbnb_lightgbm'
n_sample     = -1




####### y normalization #############################################################   
def y_norm(y, inverse=True, mode='boxcox'):
	## Normalize the input/output
	if mode == 'boxcox':
		width0 = 53.0  # 0,1 factor
		k1 = 0.6145279599674994  # Optimal boxCox lambda for y
		if inverse:
				y2 = y * width0
				y2 = ((y2 * k1) + 1) ** (1 / k1)
				return y2
		else:
				y1 = (y ** k1 - 1) / k1
				y1 = y1 / width0
				return y1

	if mode == 'norm':
		m0, width0 = 0.0, 350.0  ## Min, Max
		if inverse:
				y1 = (y * width0 + m0)
				return y1

		else:
				y2 = (y - m0) / width0
				return y2
	else:
			return y



####################################################################################
##### Params########################################################################  

"""
colnum = [  "review_scores_communication", "review_scores_location", "review_scores_rating"         ]
colcat = [ "cancellation_policy", "host_response_rate", "host_response_time" ]
coltext = ["house_rules", "neighborhood_overview", "notes", "street"  ]
coldate = [  "calendar_last_scraped", "first_review", "host_since" ]



"""
cols_input_type = {
     "coly"   : "price"
    ,"colid"  : "id"
    ,"colcat" : [  "host_id", "host_location", "host_response_time","host_response_rate","host_is_superhost","host_neighbourhood","host_verifications","host_has_profile_pic","host_identity_verified","street","neighbourhood","neighbourhood_cleansed", "neighbourhood_group_cleansed","city","zipcode", "smart_location","is_location_exact","property_type","room_type", "accommodates","bathrooms","bedrooms", "beds","bed_type","guests_included","calendar_updated", "license","instant_bookable","cancellation_policy","require_guest_profile_picture","require_guest_phone_verification","scrape_id"]
    ,"colnum" : [ "host_listings_count","latitude", "longitude","square_feet","weekly_price","monthly_price", "security_deposit","cleaning_fee","extra_people", "minimum_nights","maximum_nights","availability_30","availability_60","availability_90","availability_365","number_of_reviews","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication", "review_scores_location","review_scores_value","calculated_host_listings_count","reviews_per_month"]    
    ,"coltext" : ["name","summary", "space","description", "neighborhood_overview","notes","transit", "access","interaction", "house_rules","host_name","host_about","amenities"]
    , "coldate" : ["last_scraped","host_since","first_review","last_review"]
    ,"colcross" : ["name","host_is_superhost","is_location_exact","monthly_price","review_scores_value","review_scores_rating","reviews_per_month"]
	,"usdpricescol":["price","weekly_price","monthly_price","security_deposit","cleaning_fee","extra_people"]
    
}


#global path_config_model, path_model, path_data_train, path_data_test, path_output_pred, n_sample,model_name

def airbnb_elasticnetcv(path_model_out=""):

	global model_name
	model_name        = 'ElasticNetCV'

	def post_process_fun(y):
		return y_norm(y, inverse=True, mode='boxcox')

	def pre_process_fun(y):
		return y_norm(y, inverse=False, mode='boxcox')

	model_dict = {'model_pars': {'config_model_name': 'ElasticNetCV'
		, 'model_path': path_model_out
		, 'model_pars': {}  # default ones
		, 'post_process_fun': post_process_fun
		, 'pre_process_pars': {'y_norm_fun' : pre_process_fun,

						### Pipeline for data processing.
					   'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot',
                                        'dfdate', 'dftext'

					    ]
													 }
														 },
	'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
									  'explained_variance_score', 'r2_score', 'median_absolute_error']
									},
	'data_pars': {
			'cols_input_type' : {
								 "coly"   :   "price"
								,"colid"  :   "id"
								,"colcat" :   [ "cancellation_policy", "host_response_rate", "host_response_time" ]
								,"colnum" :   [ "review_scores_communication", "review_scores_location", "review_scores_rating"         ]
								,"coltext" :  [ "house_rules", "neighborhood_overview", "notes", "street"  ]
								,"coldate" :  [ "last_scraped", "first_review", "host_since" ]
								,"colcross" : [  ]
							 },

			'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]
		 ,'cols_model': []  # cols['colcat_model'],
		 ,'coly': []        # cols['coly']
		 ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
							}}
	return model_dict


def airbnb_lightgbm(path_model_out="") :
	"""
		Huber Loss includes L1  regurarlization
		We test different features combinaison, default params is optimal
	"""
	global model_name
	model_name        = 'LGBMRegressor'

	def post_process_fun(y):
		return y_norm(y, inverse=True, mode='boxcox')

	def pre_process_fun(y):
		return y_norm(y, inverse=False, mode='boxcox')

	model_dict = {'model_pars': {'config_model_name': 'LGBMRegressor'
		,'model_path': path_model_out
		,'model_pars': {'objective': 'huber', }  # default
		,'post_process_fun': post_process_fun
		,'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,

						### Pipeline for data processing.
					   'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
													 }
														 },
	'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
									  'explained_variance_score', 'r2_score', 'median_absolute_error']
									},

	'data_pars': {
			'cols_input_type' : {
								 "coly"   :   "price"
								,"colid"  :   "id"
								,"colcat" :   [ "cancellation_policy", "host_response_rate", "host_response_time" ]
								,"colnum" :   [ "review_scores_communication", "review_scores_location", "review_scores_rating"         ]
								,"coltext" :  [ "house_rules", "neighborhood_overview", "notes", "street"  ]
								,"coldate" :  [ "calendar_last_scraped", "first_review", "host_since" ]
								,"colcross" : [  ]
							 }
			# cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
			,'cols_model_group': [ 'colnum', 'colcat_bin']
		 ,'cols_model': []  # cols['colcat_model'],
		 ,'coly': []        # cols['coly']
		 ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data

		 }}
	return model_dict
 

def airbnb_bayesian_pyro(path_model_out="") :
	global model_name
	model_name        = 'model_bayesian_pyro'
	def post_process_fun(y):
		return y_norm(y, inverse=True, mode='boxcox')

	def pre_process_fun(y):
		return y_norm(y, inverse=False, mode='boxcox')

	model_dict = {'model_pars': {'config_model_name': 'model_bayesian_pyro'
		, 'model_path': path_model_out
		, 'model_pars': {'input_width': 112, }  # default
		, 'post_process_fun': post_process_fun

		, 'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,

						### Pipeline for data processing.
					   'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
													 }
														 },

	'compute_pars': {'compute_pars': {'n_iter': 1200, 'learning_rate': 0.01}
								 , 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
													'explained_variance_score', 'r2_score', 'median_absolute_error']
								 , 'max_size': 1000000
								 , 'num_samples': 300
	 },
	'data_pars':  {
			'cols_input_type' : {
								 "coly"   :   "price"
								,"colid"  :   "id"
								,"colcat" :   [ "cancellation_policy", "host_response_rate", "host_response_time" ]
								,"colnum" :   [ "review_scores_communication", "review_scores_location", "review_scores_rating"         ]
								,"coltext" :  [ "house_rules", "neighborhood_overview", "notes", "street"  ]
								,"coldate" :  [ "calendar_last_scraped", "first_review", "host_since" ]
								,"colcross" : [  ]
							 }
			,'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]
		 ,'cols_model': []  # cols['colcat_model'],
		 ,'coly': []        # cols['coly']
		 ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
							}}
	return model_dict
								

def airbnb_glm( path_model_out="") :
	global model_name
	model_name        = 'TweedieRegressor'
	def post_process_fun(y):
		return y_norm(y, inverse=True, mode='norm')

	def pre_process_fun(y):
		return y_norm(y, inverse=False, mode='norm')



	model_dict = {'model_pars': {'config_model_name': 'TweedieRegressor'  # Ridge
		, 'model_path': path_model_out
		, 'model_pars': {'power': 0, 'link': 'identity'}  # default ones
		, 'pre_process_pars': {'y_norm_fun' : pre_process_fun,

						### Pipeline for data processing.
					   'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ] }
														 },
							'compute_pars': {'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
															 'explained_variance_score',  'r2_score', 'median_absolute_error']
															},
	'data_pars': {
			'cols_input_type' : {
								 "coly"   :   "price"
								,"colid"  :   "id"
								,"colcat" :   [ "cancellation_policy", "host_response_rate", "host_response_time" ]
								,"colnum" :   [ "review_scores_communication", "review_scores_location", "review_scores_rating"         ]
								,"coltext" :  [ "house_rules", "neighborhood_overview", "notes", "street"  ]
								,"coldate" :  [ "calendar_last_scraped", "first_review", "host_since" ]
								,"colcross" : [  ]
							 }
			,'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]
		 ,'cols_model': []  # cols['colcat_model'],
		 ,'coly': []        # cols['coly']
		 ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
								}
	}
	return model_dict
							 


###### Path ################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)



####################################################################################################
########## Init variable ###########################################################################
globals()[config_name]()


path_config_model = root + f"/{config_file}"
path_model        = f'data/output/{data_name}/a01_{model_name}/'
path_data_train   = f'data/input/{data_name}/train/'
path_data_test    = f'data/input/{data_name}/test/'
path_output_pred  = f'/data/output/{data_name}/pred_a01_{config_name}/'


###################################################################################
########## Preprocess #############################################################
def preprocess():
	from source import run_preprocess

	run_preprocess.run_preprocess(model_name        =  config_name,
																path_data         =  path_data_train,
																path_output       =  path_model,
																path_config_model =  path_config_model,
																n_sample          =  n_sample,
																mode              =  'run_preprocess')

############################################################################
########## Train ###########################################################
def train():
	from source import run_train

	run_train.run_train(config_model_name =  config_name,
											path_data         =  path_data_train,
											path_output       =  path_model,
											path_config_model =  path_config_model , n_sample = n_sample)


###################################################################################
######### Check model #############################################################
def check():
	try :
		#### Load model
		from source.util_feature import load
		from source.models import model_sklearn as modelx
		import sys
		from source import models
		sys.modules['models'] = models

		dir_model    = path_model
		modelx.model = load( dir_model + "/model/model.pkl" )
		stats        = load( dir_model + "/model/info.pkl" )
		colsX        = load( dir_model + "/model/colsX.pkl"   )
		coly         = load( dir_model + "/model/coly.pkl"   )
		print(stats)
		print(modelx.model.model)

		### Metrics on test data
		stats['metrics_test']

		#### Loading training data  #######################################################
		ddfX     = pd.read_csv(dir_model + "/check/dfX.csv")  #to load csv
		#dfX = pd.read_parquet(dir_model + "/check/dfX.parquet")    #to load parquet
		dfy     = dfX[coly]
		colused = colsX

		dfXtest = pd.read_csv(dir_model + "/check/dfXtest.csv")    #to load csv
		#dfXtest = pd.read_parquet(dir_model + "/check/dfXtest.parquet"    #to load parquet
		dfytest = dfXtest[coly]
		print(dfX.shape,  dfXtest.shape )


		#### Feature importance on training data
		lgb_featimpt_train,_ = util_feature.feature_importance_perm(modelx, dfX[colused], dfy, colused, n_repeats=1,
																																scoring='neg_root_mean_squared_error' )

		print(lgb_featimpt_train)
	except :
			pass
	#! python source/run_inference.py  run_predict  --config_model_name  LGBMRegressor  --n_sample 1000   --path_model /data/output/a01_lightgbm_huber/    --path_output /data/output/pred_a01_lightgbm_huber/    --path_data /data/input/train/



########################################################################################
####### Inference ######################################################################
def predict():
	from source import run_inference
	run_inference.run_predict(model_name,
							path_model  = path_model,
							path_data   = path_data_test,
							path_output = path_output_pred,
							n_sample    = n_sample)


def run_all():
	preprocess()
	train()
	check()
	predict()



###########################################################################################################
###########################################################################################################
"""
python  airbnb_regression.py  preprocess
python  airbnb_regression.py  train
python  airbnb_regression.py  check
python  airbnb_regression.py  predict
python  airbnb_regression.py  run_all
"""
if __name__ == "__main__":
		import fire
		fire.Fire()
		
"""


import template_run
template_run.config_name       = config_name
template_run.path_config_model = path_config_model
template_run.path_model        = path_model
template_run.path_data_train   = path_data_train
template_run.path_data_test    = path_data_test
template_run.path_output_pred  = path_output_pred
template_run.n_sample          = n_sample
template_run.model_name        = model_name

print( template_run.config_name )
train                          = template_run.train
predict                        = template_run.predict
run_all                        = template_run.run_all
"""
