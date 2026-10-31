#generating synthetic house data
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n = 1000

df = pd.DataFrame({
    "size": rng.integers(500, 3000, n),
    "bedrooms": rng.integers(1, 6, n),
    "bathrooms": rng.integers(1, 4, n),
    "age": rng.integers(0, 40, n),
    "distance": rng.uniform(1, 25, n),
    "neighborhood": rng.choice(
        ["Premium", "Average", "Budget"],
        n,
        p=[0.25, 0.5, 0.25]
    )
})

neighborhood_effect = df["neighborhood"].map({
    "Premium": 20,
    "Average": 8,
    "Budget": -5
})

noise = rng.normal(0, 8, n)

df["price"] = (
    60
    + 0.035 * df["size"]
    + 3 * df["bedrooms"]
    + 2 * df["bathrooms"]
    - 0.5 * df["age"]
    - 1.2 * df["distance"]
    + neighborhood_effect
    + noise
)

print(df.head())
print(df.describe())


#exploring the data
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())


#visualize house size vs price
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(
    data=df,
    x="size",
    y="price",
    hue="neighborhood"
)

plt.title("House Size vs House Price")
plt.show()

#separting features and target
X = df.drop(columns=["price"])
y = df["price"]

#split test and train data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


#preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = [
    "size",
    "bedrooms",
    "bathrooms",
    "age",
    "distance"
]

categorical_features = ["neighborhood"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


#train linear regression model
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)


#make prediction
predictions = model.predict(X_test)

print("Actual prices:")
print(y_test.head())

print("Predicted prices:")
print(predictions[:5])

#model eveluation
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


#prediction of completely new house
new_house = pd.DataFrame({
    "size": [1500],
    "bedrooms": [3],
    "bathrooms": [2],
    "age": [5],
    "distance": [4],
    "neighborhood": ["Premium"]
})

predicted_price = model.predict(new_house)

print("Estimated house price:", predicted_price[0], "lakh")

df_results = pd.DataFrame({
    'Actual': y_test.values.flatten(),
    'predicted': predictions.flatten()
})
print(df_results.head())

sns.set_theme(style='whitegrid')
plt.figure(figsize=(8,6))
sns.scatterplot(x='Actual',y='predicted',data=df_results,alpha=0.7,color='b',label='Predictions')
plt.tight_layout()
plt.show()