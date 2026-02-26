import streamlit as st
import pandas as pd
import numpy as np
import math
st.title("🌳 Decision Tree Classifier")
data = pd.DataFrame({
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain',
                'Overcast','Sunny','Sunny','Rain','Sunny','Overcast',
                'Overcast','Rain'],
    'Humidity': ['High','High','High','High','Normal','Normal',
                 'Normal','High','Normal','High','Normal','High',
                 'Normal','High'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No',
                   'Yes','No','Yes','Yes','Yes','Yes',
                   'Yes','No']
})
def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    probs = counts / len(col)
    return -sum(p * math.log2(p) for p in probs)
def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[attribute], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / len(df)) *
        entropy(df[df[attribute] == values[i]][target])
        for i in range(len(values))
    )
    return total_entropy - weighted_entropy
def id3(df, target, attributes):
    if len(np.unique(df[target])) == 1:
        return df[target].iloc[0]
    if not attributes:
        return df[target].mode()[0]
    gains = [information_gain(df, attr, target) for attr in attributes]
    best_attr = attributes[np.argmax(gains)]
    tree = {best_attr: {}}
    for value in np.unique(df[best_attr]):
        subset = df[df[best_attr] == value]
        remaining = [a for a in attributes if a != best_attr]
        tree[best_attr][value] = id3(subset, target, remaining)
    return tree
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    return predict(tree[attr].get(sample[attr], "Unknown"), sample)
attributes = list(data.columns)
attributes.remove('PlayTennis')
decision_tree = id3(data, 'PlayTennis', attributes)
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Dataset")
    st.dataframe(data)
    if st.button("Generate Decision Tree"):
        st.subheader("🌲 Decision Tree")
        st.json(decision_tree)
with col2:
    st.subheader("🔮 Prediction")
    outlook = st.selectbox("Outlook", ["Sunny", "Overcast", "Rain"])
    humidity = st.selectbox("Humidity", ["High", "Normal"])
    if st.button("Predict"):
        sample = {"Outlook": outlook, "Humidity": humidity}
        result = predict(decision_tree, sample)
        st.markdown("### Result")
        if result == "Yes":
            st.success(f"✅ Play Tennis: {result}")
        elif result == "No":
            st.error(f"❌ Play Tennis: {result}")
        else:
            st.warning(f"❓ Play Tennis: {result}")
        st.write("**Sample:**", sample)
