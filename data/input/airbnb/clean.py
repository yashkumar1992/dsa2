import pandas as pd
import random


folder     = 'raw/'
df         = pd.read_csv(folder+'listings_summary.zip', delimiter=',')
df_list    = pd.read_csv(folder+'listings.zip', delimiter=',')
df_rev_sum = pd.read_csv(  folder+'reviews_summary.zip', delimiter=',')




colid = "id"
coly  = "price"



########################################################################################################
df = df.set_index(colid)
colsX = list(df.columns)
colsX.remove(coly)

df = df.sample(frac=1.0)

print(df.head(2).T)


itrain = int(len(df) * 0.8)
df.iloc[:itrain, :][colsX].reset_index().to_parquet( "train/features.parquet", index=False)
df.iloc[:itrain][coly].reset_index().to_parquet(  "train/target.parquet")



df[colsX].reset_index().iloc[itrain:, :].to_parquet( "test/features.parquet", index=False)
df[coly].reset_index().iloc[itrain:, :].to_parquet(  "test/target.parquet",  index=False)



def clean_prices(df, colnum):
    def clean(x):
        if isinstance(x, str) and x == x:
            x=x.replace('$', '').replace(',', '')
        return (x)
    for col in colnum:
        col_type = df.dtypes[col]
        if col_type == np.dtype(object):
            df[col]=df[col].astype(str).apply(clean)
            df[col]=df[col].replace({'None':None})
    df.fillna(value=pd.np.nan, inplace=True)
    df[colnum]=df[colnum].astype("float32")
    return df




"""
colnum = [  "review_scores_communication", "review_scores_location", "review_scores_rating"         ]
colcat = [ "cancellation_policy", "host_response_rate", "host_response_time" ]
coltext = ["house_rules", "neighborhood_overview", "notes", "street"  ]
coldate = [  "calendar_last_scraped", "first_review", "host_since" ]



"""




