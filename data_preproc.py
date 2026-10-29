# alwayes make sure to split data before processing
#then apply the transformations on training nad testing data 

import pandas as pd
import numpy as np

# 4 strategies:-
# mean- replaces missing values with column mean(Roughly symmetric numerical data without extreme outliers)
# meadian- uses column meadian(Numerical data with skew or outliers)
# most_frequent- uses most common value(Numerical data with skew or outliers)
#constant- uses a value specify(An explicit placeholder such as "Unknown")

#simple imputer
from sklearn.impute import SimpleImputer

#mean
df = pd.DataFrame({
    "age": [20, 25, np.nan, 35, 40],
    "income": [20000, 30000, 40000, np.nan, 60000]
})
imputer = SimpleImputer(strategy="mean")#creates transformer
imputer.fit(df[["age", "income"]])#calculates mean of each column using available variables
df_imputed = imputer.transform(df[["age", "income"]])#fills missing values with learned means
print(df_imputed)
print("Learned means:", imputer.statistics_)#statistics_ reveals learned values

#median
df = pd.DataFrame({
    "age": [20, 25, np.nan, 35, 40],
    "income": [20000, 22000, 25000, np.nan, 100000]
})
median_imputer = SimpleImputer(strategy="median")
median_imputer.fit_transform(df[["age", "income"]])
df_medimputed = median_imputer.transform(df[["age", "income"]])
print(df_medimputed)
print("Learned means:", median_imputer.statistics_)