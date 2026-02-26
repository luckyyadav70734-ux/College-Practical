import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

st.title("📧 Spam Email Classifier")

# Dataset
emails = [
    "Congratulations! You've won a free iPhone",
    "Claim your lottery prize now",
    "Exclusive deal just for you",
    "Act fast! Limited-time offer",
    "Click here to secure your reward",
    "Win cash prizes instantly by signing up",
    "Limited-time discount on luxury watches",
    "Get rich quick with this secret method",
    "Hello, how are you today",
    "Please find the attached report",
    "Thank you for your support",
    "The project deadline is next week",
    "Can we reschedule the meeting to tomorrow",
    "Your invoice for last month is attached",
    "Looking forward to our call later today",
    "Don't forget the team lunch tomorrow",
    "Meeting agenda has been updated",
    "Here are the notes from yesterday's discussion",
    "Please confirm your attendance for the workshop",
    "Let's finalize the budget proposal by Friday"
]

labels = [1]*8 + [0]*12  # 1 = Spam, 0 = Not Spam

# Layout
col1, col2 = st.columns(2)

# ---------- Left: Dataset & Training ----------
with col1:
    st.subheader("📊 Dataset")

    df = pd.DataFrame({
        "Email": emails,
        "Label": ["Spam" if l else "Not Spam" for l in labels]
    })
    st.dataframe(df)

    if st.button("Train Model"):
        with st.spinner("Training..."):
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
            X = vectorizer.fit_transform(emails)

            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.25, random_state=42, stratify=labels
            )

            model = LinearSVC()
            model.fit(X_train, y_train)

            accuracy = accuracy_score(y_test, model.predict(X_test))

            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.accuracy = accuracy

        st.success("✅ Model Trained!")
        st.metric("Accuracy", f"{accuracy:.2%}")

# ---------- Right: Prediction ----------
with col2:
    st.subheader("🔮 Spam Detection")

    new_email = st.text_area("Enter email text:", height=150)

    if st.button("Check"):
        if "model" not in st.session_state:
            st.warning("⚠️ Train the model first.")
        elif not new_email.strip():
            st.warning("⚠️ Enter some email text.")
        else:
            vector = st.session_state.vectorizer.transform([new_email])
            prediction = st.session_state.model.predict(vector)[0]

            if prediction:
                st.error("🚫 SPAM Email")
            else:
                st.success("✅ NOT SPAM")

            st.info(f"Model Accuracy: {st.session_state.accuracy:.2%}")
