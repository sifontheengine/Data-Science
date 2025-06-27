import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Title
st.title('Titanic Survival Predictor (Logistic Regression)')

# Load Data
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('Files/train.csv')

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Safely process 'Embarked' column
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    else:
        st.error("'Embarked' column not found in dataset.")

    # Encode 'Sex'
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Create new feature
    df['FamilySize'] = df['SibSp'] + df['Parch']
    return df

df = load_data()
    
# Show Data
if st.checkbox('Show Raw Data'):
    st.write(df.head())
    
# EDA
st.subheader('Explanatory Data Analysis')
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='Survived', hue='Sex', ax=ax1)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.countplot(data=df, x='Survived', hue='Pclass', ax=ax2)
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
sns.histplot(df['Age'], bins=30, kde=True, ax=ax3)
st.pyplot(fig3)

# Logistic Regression
st.subheader("Logistic Regression Model")

features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred): 2f}")

# 1. Prediction Interface
st.subheader("Predict Survival")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare", 0, 500, 50)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)


# Process Inputs
sex = 0 if sex == "male" else 1
family_size = sibsp + parch

# Form input DataFrame
input_df = pd.DataFrame([[pclass, sex, age, fare, family_size]], columns=features)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.write("Prediction:", "Survived" if prediction == 1 else "Did not survive")
    
# 2. Confusion Matrix
st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)

# 4. Classification Report (Optional)
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())
