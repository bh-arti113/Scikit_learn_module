import pandas as pd
import numpy as np


#custom data
df = pd.DataFrame({
    "size": [500, 800, 1000, 1200, 1500, 1800, 2000],
    "price": [20, 30, 40, 45, 55, 65, 75]
})
X = df[["size"]]
y = df["price"]


#data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  #recieve 20% for test data
    random_state=42
)
print(X_train)


#creating model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


#inspect learned paranetrs
print("Intercept:", model.intercept_) #bo
print("Coefficient:", model.coef_)  #b1
print(
    f"Price = {model.intercept_:.2f} "
    f"+ {model.coef_[0]:.4f} * Size"
)  #reconstructing the line


#prediction
predictions = model.predict(X_test)
print("Actual:", y_test.to_numpy())
print("Predicted:", predictions)


#new house cost prediction
new_house = pd.DataFrame({
    "size": [1300]
})
predicted_price = model.predict(new_house).round(2)
print("Predicted price:", predicted_price[0], "lakh")


#four metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

#residual plots
residuals = y_test - predictions

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()

#residual dsitribution
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.show()

#Q-Q plot
from scipy import stats
import matplotlib.pyplot as plt

stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Residual Q-Q Plot")
plt.show()