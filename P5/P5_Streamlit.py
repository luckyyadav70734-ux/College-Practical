import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
st.title("📈 Diabetes Progression Prediction")
data = load_diabetes()
X, y = data.data, data.target
st.write(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
st.write(f"Target range: {y.min():.1f} to {y.max():.1f}")
test_size = st.slider("Test Size %", 10, 40, 20)
if st.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
    st.session_state.mse = mean_squared_error(y_test, y_pred)
    st.session_state.r2 = r2_score(y_test, y_pred)
if "mse" in st.session_state:
    col1, col2 = st.columns(2)
    col1.metric("MSE", f"{st.session_state.mse:.2f}")
    col2.metric("R²", f"{st.session_state.r2:.2f}")
    fig1, ax1 = plt.subplots()
    ax1.scatter(st.session_state.y_test, st.session_state.y_pred)
    ax1.plot(
        [st.session_state.y_test.min(), st.session_state.y_test.max()],
        [st.session_state.y_test.min(), st.session_state.y_test.max()],
    )
    ax1.set_xlabel("True")
    ax1.set_ylabel("Predicted")
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots()
    ax2.scatter(st.session_state.X_test[:, 2], st.session_state.y_pred)
    ax2.set_xlabel("BMI")
    ax2.set_ylabel("Predicted")
    st.pyplot(fig2)
else:
    st.info("Train the model to see results")
