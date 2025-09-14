import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib

# Load the saved model (assume it's pickled; use joblib.load if saved with Joblib)

model_path = 'hotel_booking_prediction_model.sav'

try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App title and description
st.title("Hotel Booking Prediction App")
st.write("Enter the booking details below to predict if the booking will be canceled (using the pre-trained Bagging Pipeline Model).")

# Define input widgets based on typical hotel booking features
# Categorical features (using selectboxes with common values from dataset)
hotel = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"])
arrival_date_month = st.selectbox("Arrival Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
meal = st.selectbox("Meal Type", ["BB", "FB", "HB", "SC", "Undefined"])
country = st.selectbox("Country", ["PRT", "GBR", "USA", "ESP", "IRL", "FRA", "Other"])  # Add more if needed
market_segment = st.selectbox("Market Segment", ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Groups", "Complementary", "Aviation"])
distribution_channel = st.selectbox("Distribution Channel", ["Direct", "Corporate", "TA/TO", "GDS"])
reserved_room_type = st.selectbox("Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])
assigned_room_type = st.selectbox("Assigned Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"])
deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
is_repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1])

# Numerical features (using sliders with ranges from attached images/dataset)
lead_time = st.slider("Lead Time (days)", min_value=0, max_value=737, value=0)
arrival_date_year = st.slider("Arrival Year", min_value=2015, max_value=2017, value=2016)  # Typical range
arrival_date_week_number = st.slider("Arrival Week Number", min_value=1, max_value=53, value=1)
arrival_date_day_of_month = st.slider("Arrival Day of Month", min_value=1, max_value=31, value=1)
stays_in_weekend_nights = st.slider("Stays in Weekend Nights", min_value=0, max_value=19, value=0)
stays_in_week_nights = st.slider("Stays in Week Nights", min_value=0, max_value=50, value=0)
adults = st.slider("Adults", min_value=0, max_value=55, value=1)
children = st.slider("Children", min_value=0, max_value=10, value=0)
babies = st.slider("Babies", min_value=0, max_value=10, value=0)
previous_cancellations = st.slider("Previous Cancellations", min_value=0, max_value=26, value=0)
previous_bookings_not_canceled = st.slider("Previous Bookings Not Canceled", min_value=0, max_value=72, value=0)
booking_changes = st.slider("Booking Changes", min_value=0, max_value=21, value=0)
days_in_waiting_list = st.slider("Days in Waiting List", min_value=0, max_value=391, value=0)
adr = st.slider("Average Daily Rate (ADR)", min_value=0.0, max_value=5400.0, value=0.0, step=0.01)
required_car_parking_spaces = st.slider("Required Car Parking Spaces", min_value=0, max_value=8, value=0)
total_of_special_requests = st.slider("Total Special Requests", min_value=0, max_value=5, value=0)

# Collect inputs into a DataFrame (must match the model's expected feature order and names)
input_data = pd.DataFrame({
    'hotel': [hotel],
    'lead_time': [lead_time],
    'arrival_date_year': [arrival_date_year],
    'arrival_date_month': [arrival_date_month],
    'arrival_date_week_number': [arrival_date_week_number],
    'arrival_date_day_of_month': [arrival_date_day_of_month],
    'stays_in_weekend_nights': [stays_in_weekend_nights],
    'stays_in_week_nights': [stays_in_week_nights],
    'adults': [adults],
    'children': [children],
    'babies': [babies],
    'meal': [meal],
    'country': [country],
    'market_segment': [market_segment],
    'distribution_channel': [distribution_channel],
    'is_repeated_guest': [is_repeated_guest],
    'previous_cancellations': [previous_cancellations],
    'previous_bookings_not_canceled': [previous_bookings_not_canceled],
    'reserved_room_type': [reserved_room_type],
    'assigned_room_type': [assigned_room_type],
    'booking_changes': [booking_changes],
    'deposit_type': [deposit_type],
    'days_in_waiting_list': [days_in_waiting_list],
    'customer_type': [customer_type],
    'adr': [adr],
    'required_car_parking_spaces': [required_car_parking_spaces],
    'total_of_special_requests': [total_of_special_requests],
    # Add any missing columns from your dataset here (e.g., 'agent', 'company' if they exist)
})

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
        
        # Display prediction (assuming 0 = Not Canceled, 1 = Canceled)
        if prediction == 0:
            st.success("Prediction: Booking is NOT likely to be canceled.")
        else:
            st.warning("Prediction: Booking is likely to be canceled.")
        
        if prob is not None:
            st.write(f"Probability of Cancellation: {prob[1]:.2%}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Optional: Display input data for debugging
if st.checkbox("Show Input Data"):
    st.write(input_data)



