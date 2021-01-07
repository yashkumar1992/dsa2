# -*- coding: utf-8 -*-
"""DeltaPy - Tabular Augmentation.ipynb

Original file is located at    https://colab.research.google.com/drive/1-uJqGeKZfJegX0TmovhsO90iasyxZYiT

## DeltaPy⁠⁠ — Tabular Data Augmentation

### Introduction

Tabular augmentation is a new experimental space that makes use of novel and traditional data generation and synthesisation techniques to improve model prediction success. It is in essence a process of modular feature engineering and observation engineering while emphasising the order of augmentation to achieve the best predicted outcome from a given information set. 
Data augmentation can be defined as any method that could increase the size or improve the quality of a dataset by generating new features or instances without the collection of additional data-points. Data augmentation is of particular importance in image classification tasks where additional data can be created by cropping, padding, or flipping existing images.
Tabular cross-sectional and time-series prediction tasks can also benefit from augmentation. Here we divide tabular augmentation into columnular and row-wise methods. Row-wise methods are further divided into extraction and data synthesisation techniques, whereas columnular methods are divided into transformation, interaction, and mapping methods.  
To take full advantage of tabular augmentation for time-series you would perform the techniques in the following order: (1) transforming, (2) interacting, (3) mapping, (4) extracting, and (5) synthesising (forthcoming). What follows is a practical example of how the above methodology can be used. The purpose here is to establish a framework for table augmentation and to point and guide the user to existing packages.
See the [Skeleton Example](#example), for a combination of multiple methods that lead to a halfing of the mean squared error.

Test sets should ideally not be preprocessed with the training data, as in such a way one could be peaking ahead in the training data. The preprocessing parameters should be identified on the test set and then applied on the test set, i.e., the test set should not have an impact on the transformation applied. As an example, you would learn the parameters of PCA decomposition on the training set and then apply the parameters to both the train and the test set. 
The benefit of pipelines become clear when one wants to apply multiple augmentation methods. It makes it easy to learn the parameters and then apply them widely. For the most part, this notebook does not concern itself with 'peaking ahead' or pipelines, for some functions, one might have to restructure to code and make use of open source pacakages to create your preferred solution.


**Notebook Dependencies**
!pip install deltapy

!pip install pykalman
!pip install tsaug
!pip install ta
!pip install tsaug
!pip install pandasvault
!pip install gplearn
!pip install ta
!pip install seasonal
!pip install pandasvault


"""

"""Some of these categories are fluid and some techniques could fit into multiple buckets. 
This is an attempt to find an exhaustive number of techniques, but not an exhuastive list of implementations of the techniques. 
For example, there are thousands of ways to smooth a time-series, but we have only includes 1-2 techniques of interest under each category.

### **(1) [<font color="black">Transformation:</font>](#transformation)**
-----------------
1. Scaling/Normalisation
2. Standardisation
10. Differencing
3. Capping
13. Operations
4. Smoothing
5. Decomposing
6. Filtering
7. Spectral Analysis
8. Waveforms
9. Modifications
11. Rolling
12. Lagging
14. Forecast Model

### **(2) [<font color="black">Interaction:</font>](#interaction)**
-----------------
1. Regressions
2. Operators
3. Discretising
4. Normalising
5. Distance
6. Speciality
7. Genetic

### **(3) [<font color="black">Mapping:</font>](#mapping)**
-----------------
1. Eigen Decomposition
2. Cross Decomposition
3. Kernel Approximation
4. Autoencoder
5. Manifold Learning
6. Clustering
7. Neighbouring

### **(4) [<font color="black">Extraction:</font>](#extraction)**
-----------------
1. Energy
2. Distance
3. Differencing
4. Derivative
5. Volatility
6. Shape
7. Occurence
8. Autocorrelation
9. Stochasticity
10. Averages
11. Size
13. Count
14. Streaks
14. Location
15. Model Coefficients
16. Quantile
17. Peaks
18. Density
20. Linearity
20. Non-linearity
21. Entropy
22. Fixed Points
23. Amplitude
23. Probability
24. Crossings
25. Fluctuation
26. Information
27. Fractals
29. Exponent
30. Spectral Analysis
31. Percentile
32. Range
33. Structural
12. Distribution


"""
import warnings, os, sys
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from deltapy import transform, interact, mapper, # extract

try :
  import pandasvault

except:
  pass



###########################################################################################
###########################################################################################
def data_copy():
  df = pd.read_csv("https://github.com/firmai/random-assets-two/raw/master/numpy/tsla.csv")
  df["Close_1"] = df["Close"].shift(-1)
  df = df.dropna()
  df["Date"] = pd.to_datetime(df["Date"])
  df = df.set_index("Date")
  return df

def test_prepro_all():
  df = data_copy(); df.head()

  df_out = transform.robust_scaler(df.copy(), drop=["Close_1"]); df_out.head()
  df_out = transform.standard_scaler(df.copy(), drop=["Close"]); df_out.head()           
  df_out = transform.fast_fracdiff(df.copy(), ["Close","Open"],0.5); df_out.head()
  df_out = transform.windsorization(df.copy(),"Close",para,strategy='both'); df_out.head()
  df_out = transform.operations(df.copy(),["Close"]); df_out.head()
  df_out= transform.triple_exponential_smoothing(df.copy(),["Close"], 12, .2,.2,.2,0); 
  df_out = transform.naive_dec(df.copy(), ["Close","Open"]); df_out.head()
  df_out = transform.bkb(df.copy(), ["Close"]); df_out.head()
  df_out = transform.butter_lowpass_filter(df.copy(),["Close"],4); df_out.head()
  df_out = transform.instantaneous_phases(df.copy(), ["Close"]); df_out.head()
  df_out = transform.kalman_feat(df.copy(), ["Close"]); df_out.head()
  df_out = transform.perd_feat(df.copy(),["Close"]); df_out.head()
  df_out = transform.fft_feat(df.copy(), ["Close"]); df_out.head()
  df_out = transform.harmonicradar_cw(df.copy(), ["Close"],0.3,0.2); df_out.head()
  df_out = transform.saw(df.copy(),["Close","Open"]); df_out.head()
  df_out = transform.modify(df.copy(),["Close"]); df_out.head()
  df_out = transform.multiple_rolling(df, columns=["Close"]); df_out.head()
  df_out = transform.multiple_lags(df, start=1, end=3, columns=["Close"]); df_out.head()
  df_out  = transform.prophet_feat(df.copy().reset_index(),["Close","Open"],"Date", "D"); df_out.head()

  #**Interaction**
  df_out = interact.lowess(df.copy(), ["Open","Volume"], df["Close"], f=0.25, iter=3); df_out.head()
  df_out = interact.autoregression(df.copy()); df_out.head()
  df_out = interact.muldiv(df.copy(), ["Close","Open"]); df_out.head()
  df_out = interact.decision_tree_disc(df.copy(), ["Close"]); df_out.head()
  df_out = interact.quantile_normalize(df.copy(), drop=["Close"]); df_out.head()
  df_out = interact.tech(df.copy()); df_out.head()
  df_out = interact.genetic_feat(df.copy()); df_out.head()

  #**Mapping**
  df_out = mapper.pca_feature(df.copy(),variance_or_components=0.80,drop_cols=["Close_1"]); df_out.head()
  df_out = mapper.cross_lag(df.copy()); df_out.head()
  df_out = mapper.a_chi(df.copy()); df_out.head()
  df_out = mapper.encoder_dataset(df.copy(), ["Close_1"], 15); df_out.head()
  df_out = mapper.lle_feat(df.copy(),["Close_1"],4); df_out.head()
  df_out = mapper.feature_agg(df.copy(),["Close_1"],4 ); df_out.head()
  df_out = mapper.neigh_feat(df.copy(),["Close_1"],4 ); df_out.head()


  #**Extraction**
  extract.abs_energy(df["Close"])
  extract.cid_ce(df["Close"], True)
  extract.mean_abs_change(df["Close"])
  extract.mean_second_derivative_central(df["Close"])
  extract.variance_larger_than_standard_deviation(df["Close"])
  extract.var_index(df["Close"].values,var_index_param)
  extract.symmetry_looking(df["Close"])
  extract.has_duplicate_max(df["Close"])
  extract.partial_autocorrelation(df["Close"])
  extract.augmented_dickey_fuller(df["Close"])
  extract.gskew(df["Close"])
  extract.stetson_mean(df["Close"])
  extract.length(df["Close"])
  extract.count_above_mean(df["Close"])
  extract.longest_strike_below_mean(df["Close"])
  extract.wozniak(df["Close"])
  extract.last_location_of_maximum(df["Close"])
  extract.fft_coefficient(df["Close"])
  extract.ar_coefficient(df["Close"])
  extract.index_mass_quantile(df["Close"])
  extract.number_cwt_peaks(df["Close"])
  extract.spkt_welch_density(df["Close"])
  extract.linear_trend_timewise(df["Close"])
  extract.c3(df["Close"])
  extract.binned_entropy(df["Close"])
  extract.svd_entropy(df["Close"].values)
  extract.hjorth_complexity(df["Close"])
  extract.max_langevin_fixed_point(df["Close"])
  extract.percent_amplitude(df["Close"])
  extract.cad_prob(df["Close"])
  extract.zero_crossing_derivative(df["Close"])
  extract.detrended_fluctuation_analysis(df["Close"])
  extract.fisher_information(df["Close"])
  extract.higuchi_fractal_dimension(df["Close"])
  extract.petrosian_fractal_dimension(df["Close"])
  extract.hurst_exponent(df["Close"])
  extract.largest_lyauponov_exponent(df["Close"])
  extract.whelch_method(df["Close"])
  extract.find_freq(df["Close"])
  extract.flux_perc(df["Close"])
  extract.range_cum_s(df["Close"])
  extract.structure_func(df["Close"])
  extract.kurtosis(df["Close"])
  extract.stetson_k(df["Close"])



