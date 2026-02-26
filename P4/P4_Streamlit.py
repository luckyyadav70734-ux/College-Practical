import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("🎬 IMDB Sentiment Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Upload Dataset")
    file = st.file_uploader("Upload CSV (review, sentiment)", type="csv")

    if file:
        try:
            df = pd.read_csv(file)
            if not {"review", "sentiment"}.issubset(df.columns):
                st.error("CSV must contain 'review' and 'sentiment'")
                st.stop()

            df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

            st.success("File uploaded")
            st.write("Total:", len(df))
            st.write("Positive:", (df["sentiment"] == 1).sum())
            st.write("Negative:", (df["sentiment"] == 0).sum())
            st.dataframe(df.head())

            max_features = st.slider("Max Features", 1000, 10000, 5000, 1000)
            test_size = st.slider("Test Size %", 10, 40, 20)

            if st.button("Train Model"):
                X_train, X_test, y_train, y_test = train_test_split(
                    df["review"], df["sentiment"],
                    test_size=test_size/100, random_state=42, stratify=df["sentiment"]
                )

                tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
                X_train = tfidf.fit_transform(X_train)
                X_test = tfidf.transform(X_test)
                model = MultinomialNB().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.session_state.model = model
                st.session_state.tfidf = tfidf
                st.session_state.accuracy = accuracy_score(y_test, y_pred)
                st.session_state.report = classification_report(y_test, y_pred)
                st.session_state.cm = confusion_matrix(y_test, y_pred)
                st.metric("Accuracy", f"{st.session_state.accuracy:.2%}")
        except:
            st.error("Error loading file")
with col2:
    st.subheader("📈 Results")
    if "accuracy" in st.session_state:
        st.text(st.session_state.report)
        fig, ax = plt.subplots()
        sns.heatmap(st.session_state.cm, annot=True, fmt="d",
                    xticklabels=["Negative","Positive"],
                    yticklabels=["Negative","Positive"], ax=ax)
        st.pyplot(fig)
    st.subheader("🔮 Predict Sentiment")
    text = st.text_area("Enter review")
    if st.button("Analyze"):
        if "model" not in st.session_state:
            st.warning("Train model first")
        elif not text.strip():
            st.warning("Enter review text")
        else:
            vec = st.session_state.tfidf.transform([text])
            pred = st.session_state.model.predict(vec)[0]
            if pred == 1:
                st.success("😊 Positive")
            else:
                st.error("😞 Negative")
            st.info(f"Accuracy: {st.session_state.accuracy:.2%}")
with st.sidebar:
    example = pd.DataFrame({
        "review": ["Great movie!", "Bad acting"],
        "sentiment": ["positive", "negative"]
    })
    st.download_button("Download Example CSV",
                       example.to_csv(index=False),
                       "imdb_example.csv",
                       "text/csv")
