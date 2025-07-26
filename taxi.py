import streamlit as st
import numpy as np
import pickle

# Load your trained model
with open(r"C:\Users\HOME\Desktop\taxi project\best_model_GBR.pkl", "rb") as fr:
    model = pickle.load(fr)

def main():
    st.title("Hello, Streamlit!")
    st.write("This is your first Streamlit web app.")

    st.sidebar.header("Uber Fare Cost Prediction")
    st.sidebar.text("This web app predicts Uber fare cost.")
    st.sidebar.header("Just fill in the information below")

    st.text("The GradientBoostingRegressor model was used.")

    # Input sliders
    pickup_longitude = st.slider("Input Your Pickup Longitude", -180.0, 180.0)
    pickup_latitude = st.slider("Input your Pickup Latitude", -90.0, 90.0)
    dropoff_longitude = st.slider("Input your Dropoff Longitude", -180.0, 180.0)
    dropoff_latitude = st.slider("Input your Dropoff Latitude", -90.0, 90.0)
    passenger_count = st.slider("Input your Passenger Count", 1, 8)

    # Prepare input for the model
    inputs = [[pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count]]

    # Predict button
    if st.button('Predict'):
        result = model.predict(inputs)
        updated_res = result.flatten().astype(float)
        st.success('Your predicted fare will be ₹{:.2f}'.format(updated_res[0]))

# Run main
if __name__ == '__main__':
    main()
