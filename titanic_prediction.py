# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load Dataset

df = pd.read_csv("Titanic-Dataset.csv")


# Display First 5 Rows

print("First 5 Rows:")
print(df.head())


# Dataset Shape

print("\nDataset Shape:")
print(df.shape)


# Column Names

print("\nColumn Names:")
print(df.columns)


# Dataset Information

print("\nDataset Information:")
df.info()


# Statistical Summary

print("\nStatistical Summary:")
print(df.describe())


# Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())


# Data Cleaning

# Remove Cabin column

if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)


# Fill missing Age values with mean

df["Age"] = df["Age"].fillna(df["Age"].mean())


# Fill missing Embarked values with mode

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# Check Missing Values After Cleaning

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# Exploratory Data Analysis (EDA)


# Survival Count

plt.figure(figsize=(6,4))

sns.countplot(x="Survived", data=df)

plt.title("Survival Count")

plt.show()



# Survival by Gender

plt.figure(figsize=(6,4))

sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Survival by Gender")

plt.show()



# Survival by Passenger Class

plt.figure(figsize=(6,4))

sns.countplot(x="Pclass", hue="Survived", data=df)

plt.title("Survival by Passenger Class")

plt.show()
# Encoding Categorical Columns

# Convert Sex column
df["Sex"] = df["Sex"].map({"male":0, "female":1})


# Convert Embarked column
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)


# Check Dataset
print("\nDataset After Encoding:")
print(df.head())
# Feature Selection

X = df.drop(["Survived", "Name", "Ticket"], axis=1)

y = df["Survived"]


print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
from sklearn.linear_model import LogisticRegression


model = LogisticRegression(max_iter=1000)


model.fit(X_train, y_train)


print("Model Training Completed")
# Predictions

y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
import pickle

with open("titanic_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully")