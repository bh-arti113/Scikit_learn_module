import pandas as pd
# loading a built in dataset
from sklearn.datasets import load_iris

#will split the data using this
from sklearn.model_selection import train_test_split


#logistic regression model
#we imported the model which we will use, in this case i.e logistic regression(as it's a classification problem)
from sklearn.linear_model import LogisticRegression

#to evluate the model we import accuracy_score,classification_report
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris(as_frame=True) #converts dataset into pandas DataFrame format

X = iris.data
y = iris.target

print(X.head())
print(y.head())

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Classes:", iris.target_names)


# spliting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #it assigns 20% of total data to test set
    random_state=42, #make sure that the split is reproducible
    stratify=y  #make sure that the final set have proportionate classes
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#training the model
model = LogisticRegression(max_iter=1000) #default max_iter is 100, we increase it so model can find the optimal answer in suuficient steps and they arent less

model.fit(X_train, y_train) #passed the training set in the model

#make the prediction
predictions = model.predict(X_test) #passed the test set in the model,it predicts values for the test set

print("Predictions lr:", predictions[:10])
print("Actual:", y_test.to_numpy()[:10]) #.to_numpy() to convert pd dataframe into numpy array

#evaluate model
lr_accuracy = accuracy_score(y_test, predictions)
print("Test accuracy:", lr_accuracy)
print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))


#decision tree classifier model
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)  #train model

dt_predictions = dt_model.predict(X_test)#prediction
print("Predictions dt: ",dt_predictions[:10])

dt_accuracy = accuracy_score(y_test, dt_predictions) #model evaluation
print("Decision Tree Accuracy:", dt_accuracy)

#model comparision
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [lr_accuracy, dt_accuracy]
})

print(results.sort_values("Accuracy", ascending=False))