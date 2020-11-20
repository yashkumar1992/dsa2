


df = pd.read_csv(folder+'listings_summary.zip', delimiter=',')

df_list = pd.read_csv(folder+'listings.zip', delimiter=',')
df_rev_sum = pd.read_csv(  folder+'reviews_summary.zip', delimiter=',')





Index(['listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space',
        'description', 'experiences_offered', 'neighborhood_overview', 'notes',
        'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url',
        'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url',
        'host_name', 'host_since', 'host_location', 'host_about',
        'host_response_time', 'host_response_rate', 'host_acceptance_rate',
        'host_is_superhost', 'host_thumbnail_url', 'host_picture_url',
        'host_neighbourhood', 'host_listings_count',
        'host_total_listings_count', 'host_verifications',
        'host_has_profile_pic', 'host_identity_verified', 'street',
        'neighbourhood', 'neighbourhood_cleansed',
        'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
        'smart_location', 'country_code', 'country', 'latitude', 'longitude',
        'is_location_exact', 'property_type', 'room_type', 'accommodates',
        'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet',
        'price', 'weekly_price', 'monthly_price', 'security_deposit',
        'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
        'maximum_nights', 'calendar_updated', 'has_availability',
        'availability_30', 'availability_60', 'availability_90',
        'availability_365', 'calendar_last_scraped', 'number_of_reviews',
        'first_review', 'last_review', 'review_scores_rating',
        'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value', 'requires_license',
        'license', 'jurisdiction_names', 'instant_bookable',
        'is_business_travel_ready', 'cancellation_policy',
        'require_guest_profile_picture', 'require_guest_phone_verification',
        'calculated_host_listings_count', 'reviews_per_month'],


colid = "id"
colnum = [  "review_scores_communication", "review_scores_location", "review_scores_rating"
colcat = [ "cancellation_policy", "host_response_rate", "host_response_time" ]
coltext = ["house_rules", "neighborhood_overview", "notes", "street"  ]
coldate = [  "calendar_last_scraped", "first_review", "host_since" ]
coly = "price"
colall = colnum + colcat + coltext + coldate




### Remoe common words
import json
import string
punctuations = string.punctuation


stopwords = json.load(open("stopwords_en.json") )["word"]
stopwords = [ t for t in string.punctuation ] + stopwords
stopwords = [ "", " ", ",", ".", "-", "*", '€', "+", "/" ] + stopwords
stopwords =list(set( stopwords ))
stopwords.sort()
print( stopwords )


stopwords = set(stopwords)



#### NA to ""
df[coltext] =  df[coltext].fillna("")
print( df[coltext].isnull().sum() )


# In[371]:



dftext = util_text.pd_coltext_clean( df[coltext], coltext, stopwords= stopwords ) 
        
dftext.head(6)



### Word Token List
coltext_freq = {}
for col in coltext :
     coltext_freq[col] =  util_text.pd_coltext_wordfreq(dftext, col) 
    
coltext_freq





# In[374]:


print(coltext_freq["house_rules"].values[:10])


# In[396]:


ntoken=100
dftext_tdidf_dict, word_tokeep_dict = {}, {}
    
for col in coltext:
   word_tokeep = coltext_freq[col]["word"].values[:ntoken]
   word_tokeep = [  t for t in word_tokeep if t not in stopwords   ]
 
   dftext_tdidf_dict[col], word_tokeep_dict[col] = util_text.pd_coltext_tdidf( dftext, coltext= col,  word_minfreq= 1,
                                                        word_tokeep= word_tokeep ,
                                                        return_val= "dataframe,param"  )
dftext_tdidf_dict, word_tokeep_dict




###  Dimesnion reduction for Sparse Matrix
dftext_svd_list, svd_list  = {},{}
for col in  coltext :
    dftext_svd_list[col], svd_list[col] = util_model.pd_dim_reduction(dftext_tdidf_dict[col], 
                                               colname=None,
                                               model_pretrain=None,                       
                                               colprefix= col + "_svd",
                                               method="svd", 
                                                           dimpca=2, 
                                                           return_val="dataframe,param")



dftext_svd_list, svd_list  
    



## To pipeline
#  Save each feature processing into "Reproductible pipeline".
#  For Scalability, for repreoduction process

######### Pipeline ONE
col = 'house_rules'


pipe_preprocess_coltext01 =[ 
           ( util_text.pd_coltext_clean , {"colname": [col], "stopwords"  : stopwords },  )        

          ,( util_text.pd_coltext_tdidf, { "coltext": col,  "word_minfreq" : 1,
                                          "word_tokeep" :  word_tokeep_dict[col],
                                          "return_val": "dataframe"  } , "convert to TD-IDF vector")
    

          ,( util_model.pd_dim_reduction, { "colname": None, 
                                          "model_pretrain" : svd_list[col],
                                          "colprefix": col + "_svd",
                                          "method": "svd", "dimpca" :2, 
                                          "return_val": "dataframe"  } , "Dimension reduction")        
]

### Check pipeline
print( col, word_tokeep )
util_feature.pd_pipeline_apply( df[[col ]].iloc[:10,:], pipe_preprocess_coltext01).iloc[:10, :]  



######### Pipeline TWO
ntoken= 100
col =  'neighborhood_overview'
pipe_preprocess_coltext02 =[ 
           ( util_text.pd_coltext_clean , {"colname": [col], "stopwords"  : stopwords },  )        

          ,( util_text.pd_coltext_tdidf, { "coltext": col,  "word_minfreq" : 1,
                                          "word_tokeep" :  word_tokeep_dict[col],
                                          "return_val": "dataframe"  } , "convert to TD-IDF vector")
    

          ,( util_model.pd_dim_reduction, { "colname": None, 
                                          "model_pretrain" : svd_list[col],
                                          "colprefix": col + "_svd",
                                          "method": "svd", "dimpca" :2, 
                                          "return_val": "dataframe"  } , "Dimension reduction")        
]

### Check pipeline
print( col, word_tokeep )
util_feature.pd_pipeline_apply( df[[ col ]].iloc[:10, :], pipe_preprocess_coltext02).iloc[:10]  



# In[338]:


dftext[coltext]




###################################################################################################
###################################################################################################
coldate = [  "first_review", "host_since" ]

df[coldate].head(4)

import dateutil
import copy
from datetime import datetime

pd_datestring_split( df , col, fmt="auto" ).head(5)


df[coldate].iloc[ :10 , :]


dfdate_list, coldate_list  = {},{}
for col in  coldate :
    dfdate_list[col] = pd_datestring_split( df , col, fmt="auto", "return_val": "split" )
    coldate_list[col] =  [   t for t in  dfdate_list[col].columns if t not in  [col, col +"_dt"]      ]
    

dfdate_list, coldate_list


######### Pipeline ##########################################
ntoken= 100
col =  'neighborhood_overview'


pipe_preprocess_coltext02 =[ 
           ( util_text.pd_coldate_split , {"colname": col, "fmt": "auto", "return_val": "split"  },  )        
     
]

### Check pipeline
print( col, word_tokeep )
util_feature.pd_pipeline_apply( df[[ col ]].iloc[:10, :], pipe_preprocess_coltext02).iloc[:10]  




############ Hashing
dfdate_hash, coldate_hash_model= util_text.pd_coltext_minhash(df, coldate, n_component=[4, 2], 
                                                    model_pretrain_dict=None,       
                                                    return_val="dataframe,param") 
dfdate_hash, coldate_hash_model


######### Pipeline ##########################################
pipe_preprocess_coldate_01 =[ 
    (util_text.pd_coltext_fillna , {"colname": coldate, "val" : ""  },  )   ,
    
    (util_text.pd_coltext_minhash , {"colname": coldate, "n_component" : [],
                                          "model_pretrain_dict" : coldate_hash_model,
                                           "return_val": "dataframe"  },  )        
     
]
    
### Check pipeline
print( coldate )
util_feature.pd_pipeline_apply( df[ coldate ].iloc[:10, :], pipe_preprocess_coldate_01).iloc[:10]  





