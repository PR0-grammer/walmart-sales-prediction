import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("walmart_best_model.pkl")

st.title("Walmart Weekly Sales Predictor")

def user_input_features():
    Store = st.sidebar.number_input('Store ID', 1, 50, 1)
    Temperature = st.sidebar.number_input('Temperature (°C)', -30.0, 50.0, 15.0)
    Fuel_Price = st.sidebar.number_input('Fuel Price ($)', 0.0, 5.0, 3.0)
    CPI = st.sidebar.number_input('CPI', 0.0, 300.0, 175.0)
    Unemployment = st.sidebar.number_input('Unemployment (%)', 0.0, 30.0, 7.0)
    Holiday_Flag = 1 if st.sidebar.selectbox('Is Today a Holiday?', ('No', 'Yes')) == 'Yes' else 0

    # Allow a reasonable date range: from the first date in dataset up to a future range (2025)
    Date = st.sidebar.date_input('Date', value=pd.to_datetime('2012-06-15'), min_value=pd.to_datetime('2010-02-05'), max_value=pd.to_datetime('2025-12-31'))

    Day = Date.day
    Month = Date.month
    Week = Date.isocalendar()[1]
    Year = Date.year

    # Warn if the selected year is outside the training range
    if Year > 2015:
        st.warning("⚠️ Predicting beyond 2015 — model was trained on 2010–2012 data. Results may be unreliable.")

    data = {
        'Store': Store,
        'Temperature_(C)': Temperature,
        'Fuel_Price': Fuel_Price,
        'CPI': CPI,
        'Unemployment': Unemployment,
        'Holiday_Flag': Holiday_Flag,
        'Day': Day,
        'Month': Month,
        'Week': Week,
        'Year': Year
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
input_df.columns = input_df.columns.str.strip()

st.subheader('Input Features')
st.write(input_df)

if st.button('Predict Weekly Sales'):
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    if missing_cols:
        st.error(f"Missing columns for prediction: {missing_cols}")
    else:
        input_df = input_df[model.feature_names_in_]
        prediction = model.predict(input_df)
        st.subheader('Predicted Weekly Sales')
        st.metric("Predicted Weekly Sales", f"${prediction[0]:,.2f}")

st.subheader("Feature Importance")
st.image("feature_importance.png", use_container_width=True)


st.markdown("---\nCreated by AAA")
