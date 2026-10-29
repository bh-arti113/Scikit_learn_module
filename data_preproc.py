# alwayes make sure to split data before processing
#then apply the transformations on training nad testing data 

import pandas as pd
import numpy as np

#NUMERICAL DATA
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


#CATEGORICAL DATA

#One-hot encoding
from sklearn.preprocessing import OneHotEncoder
cities = pd.DataFrame({
    "city": ["Delhi", "Mumbai", "Ludhiana", "Delhi"]
})
encoder = OneHotEncoder(handle_unknown="ignore")
encoded = encoder.fit_transform(cities) 
print(encoded.toarray())

#Ordinal encoding
from sklearn.preprocessing import OrdinalEncoder
education = pd.DataFrame({
    "education": [
        "School",
        "Bachelor's",
        "Master's",
        "PhD"
    ]
})
encoder = OrdinalEncoder(
    categories=[["School", "Bachelor's", "Master's", "PhD"]]
)
print(encoder.fit_transform(education))

#Feature-Scaling
#1. Standarisation, transforms every feature z= x-u/s... x= original data, u= mean of training data, s= standard deviation of traing data
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    "age": [20, 30, 40, 50, 60],
    "income": [20000, 40000, 60000, 80000, 100000]
})
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
scaled_df = pd.DataFrame(
    scaled,
    columns=data.columns
)
print(scaled_df.round(2))

#2. Min-Max scaling, (0-1) x' = x-xmin/xmax-xmin
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
normalized = minmax_scaler.fit_transform(data)
print(pd.DataFrame(
    normalized,
    columns=data.columns
).round(2))

#Detecting outliers
ages = pd.Series([22, 25, 27, 30, 32, 250])

Q1 = ages.quantile(0.25)
Q3 = ages.quantile(0.75)

IQR = Q3 - Q1 #inter-quartile range

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ages[
    (ages < lower_bound) |
    (ages > upper_bound)
]

print("Potential outliers:", outliers.tolist())


#FINAL WORKFLOW (same type of data)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

#FINAL WORFLOW(when data columns have different types)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Example customer dataset
df = pd.DataFrame({
    "age": [25, 32, np.nan, 45, 29, 38, 41, 23, 50, 34],
    "income": [40000, 65000, 52000, np.nan, 48000,
               75000, 90000, 35000, 110000, 58000],
    "city": ["Delhi", "Mumbai", "Delhi", "Chandigarh", "Mumbai",
             "Delhi", "Chandigarh", "Mumbai", "Delhi", "Mumbai"],
    "purchased": ["No", "Yes", "No", "Yes", "No",
                  "Yes", "Yes", "No", "Yes", "No"]
})

X = df.drop(columns="purchased")
y = df["purchased"]

numeric_features = ["age", "income"]
categorical_features = ["city"]

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Apply the correct pipeline to each column group
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Join preprocessing and model training
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))