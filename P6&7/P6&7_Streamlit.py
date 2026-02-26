import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
st.title("🚢 Titanic Survival Predictor")
col1, col2 = st.columns(2)
with col1:
    file = st.file_uploader("Upload Titanic CSV", type="csv")
    if file:
        try:
            df = pd.read_csv(file)
            required = ['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked','Survived']
            if not set(required).issubset(df.columns):
                st.error("Missing required columns")
                st.stop()
            df['Age'] = df['Age'].fillna(df['Age'].median())
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            df = df.dropna(subset=['Embarked'])
            df = pd.get_dummies(df, columns=['Sex','Embarked'], drop_first=True)
            features = ['Pclass','Age','SibSp','Parch','Fare','Sex_male','Embarked_Q','Embarked_S']
            X, y = df[features], df['Survived']
            test_size = st.slider("Test Size %", 10, 40, 20)
            if st.button("Train Model"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.accuracy = accuracy_score(y_test, y_pred)
                st.session_state.report = classification_report(y_test, y_pred)
                st.session_state.cm = confusion_matrix(y_test, y_pred)
                st.metric("Accuracy", f"{st.session_state.accuracy:.3f}")
        except:
            st.error("Error loading file")
with col2:
    if "accuracy" in st.session_state:
        st.text(st.session_state.report)
        fig, ax = plt.subplots()
        sns.heatmap(st.session_state.cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)
        st.subheader("Predict Survival")
        with st.form("form"):
            pclass = st.selectbox("Class",[1,2,3])
            age = st.number_input("Age",0,100,30)
            sibsp = st.number_input("SibSp",0,10,0)
            parch = st.number_input("Parch",0,10,0)
            fare = st.number_input("Fare",0.0,600.0,50.0)
            sex = st.selectbox("Sex",["Male","Female"])
            embarked = st.selectbox("Embarked",["S","C","Q"])
            submit = st.form_submit_button("Predict")
            if submit:
                input_df = pd.DataFrame([{
                    'Pclass':pclass,
                    'Age':age,
                    'SibSp':sibsp,
                    'Parch':parch,
                    'Fare':fare,
                    'Sex_male':1 if sex=="Male" else 0,
                    'Embarked_Q':1 if embarked=="Q" else 0,
                    'Embarked_S':1 if embarked=="S" else 0
                }])
                scaled = st.session_state.scaler.transform(input_df)
                pred = st.session_state.model.predict(scaled)[0]
                prob = st.session_state.model.predict_proba(scaled)[0]
                if pred==1:
                    st.success(f"Survived ({prob[1]:.1%})")
                else:
                    st.error(f"Did Not Survive ({prob[0]:.1%})")
                st.info(f"Accuracy: {st.session_state.accuracy:.3f}")
    else:
        st.info("Upload data and train model")
