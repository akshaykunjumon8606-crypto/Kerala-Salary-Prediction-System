import streamlit as st
import pandas as pd
import pickle
with open("salary_model.pkl", "rb") as f:
    model, le_dict = pickle.load(f)
st.title(" Kerala Salary Prediction System")
st.write("Enter candidate details below:")
experience = st.slider("Years of Experience", 0, 20, 2)
education = st.selectbox(
    "Education",
    le_dict["Education"].classes_
)
location = st.selectbox(
    "Location",
    le_dict["Location"].classes_
)
role = st.selectbox(
    "Role",
    le_dict["Role"].classes_
)
company_type = st.selectbox(
    "Company Type",
    le_dict["Company_Type"].classes_
)
if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        "Experience": [experience],
        "Education": [education],
        "Location": [location],
        "Role": [role],
        "Company_Type": [company_type]
    })
    for col in ["Education", "Location", "Role", "Company_Type"]:
        input_data[col] = le_dict[col].transform(input_data[col])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹ {int(prediction):,}")
    st.subheader("Salary vs Experience Trend")
import numpy as np
exp_range = np.arange(0, 16)
sample_data = pd.DataFrame({
    "Experience": exp_range,
    "Education": le_dict["Education"].transform([education]*16),
    "Location": le_dict["Location"].transform([location]*16),
    "Role": le_dict["Role"].transform([role]*16),
    "Company_Type": le_dict["Company_Type"].transform([company_type]*16)
})
predicted_salaries = model.predict(sample_data)
chart_df = pd.DataFrame({
    "Experience": exp_range,
    "Predicted Salary": predicted_salaries
})
st.line_chart(chart_df.set_index("Experience"))





