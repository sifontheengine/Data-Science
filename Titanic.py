import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the Data
df = pd.read_csv('Files/train.csv')
print(df.head())

# 2. Clean & Understand the Data
print(df.info())
print(df.describe())
print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 3. Exploratory Data Analysis (EDA)
# Survival by gender
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title('Survival by Gender')

# Survival by passenger class
sns.countplot(data=df, x='Survived', hue='Pclass')
plt.title('Survived by Class')

# Age distribution
sns.histplot(df['Age'], bins=30, kde=True)

plt.show()

# 4. Feature Engineering
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# (Optional)
df['FamilySize'] = df['SibSp'] + df['Parch']

# 5. Basic Model (Optional Intro to ML)
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# import os
# print(os.getcwd())
