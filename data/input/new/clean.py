
"""
python  clean.py

python  clean.py  profile     #### Data Profile

python  clean.py  to_train

python  clean.py  to_test



"""
import pandas as pd, numpy as np
#######################################################################################



##### Load from samples
df = pd.read_csv( 'raw/raw.csv', nrows=5000)
print(df.head(5).T)
print(df.tail(5).T)
print(df.dtypes)



 
#######################################################################################
"""
Put manually column by data type :


"""

colid  = ""

coly   = ""  #"PassengerId"

colcat = []

colnum = []

colsX  = colcat + colnum

print('coly',  coly)
print('colsX', colsX)



#######################################################################################
def profile() :
	os.makedirs("profile/", exist_ok=True)
	for x in colcat:
	   df[x] = df[x].factorize()[0]

	##### Pandas Profile   ###################################
	profile = df.profile_report(title='Profile data')
	profile.to_file(output_file=  "profile/raw_report.html")
    print( "profile/raw_report.html" )



#######################################################################################
def create_features(df)
    return df




#######################################################################################
def to_train() :
	os.makedirs("train/", exist_ok=True)

	df = create_features(df)

	Xcols = []
	df[Xcols  ].to_parquet( "train/features.parquet"   )
	df[[ coly ]].to_parquet( "train/target.parquet"   )



#######################################################################################
def to_val() :
	os.makedirs("test/", exist_ok=True)

	df = create_features(df)

	Xcols = []
	df[ Xcols  ].to_parquet( "val/features.parquet"   )
	df[[ coly ]].to_parquet( "val/target.parquet"   )





#######################################################################################
def to_test() :
	os.makedirs("test/", exist_ok=True)

	df = create_features(df)

	Xcols = []
	df[ Xcols  ].to_parquet( "test/features.parquet"   )
	df[[ coly ]].to_parquet( "test/target.parquet"   )







"""
python  clean.py

python  clean.py  profile

python  clean.py  to_train
python  clean.py  to_test


"""
if __name__ == "__main__":
    import fire
    fire.Fire()






\