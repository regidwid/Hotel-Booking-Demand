import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved pipeline model
with open('hotel_booking_prediction.sav', 'rb') as f:
 model = pickle.load(f)


# App title and description
st.title("Hotel Booking Cancellation Prediction")
st.write("Enter the booking details below to predict if the booking will be canceled. The model includes built-in preprocessing.")

# Define input fields based on typical hotel booking features
# Numerical features with assumed min-max ranges (adjust based on your images)
lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=737, value=0)
arrival_date_year = st.number_input("Arrival Date Year", min_value=2015, max_value=2017, value=2016)
arrival_date_week_number = st.number_input("Arrival Date Week Number", min_value=1, max_value=53, value=1)
arrival_date_day_of_month = st.number_input("Arrival Date Day of Month", min_value=1, max_value=31, value=1)
stays_in_weekend_nights = st.number_input("Stays in Weekend Nights", min_value=0, max_value=19, value=0)
stays_in_week_nights = st.number_input("Stays in Week Nights", min_value=0, max_value=50, value=1)
adults = st.number_input("Number of Adults", min_value=0, max_value=55, value=1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
babies = st.number_input("Number of Babies", min_value=0, max_value=10, value=0)
previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=26, value=0)
previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=72, value=0)
booking_changes = st.number_input("Booking Changes", min_value=0, max_value=21, value=0)
days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=391, value=0)
adr = st.number_input("ADR (Average Daily Rate)", min_value=0.0, max_value=5400.0, value=100.0, step=0.01)
required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, max_value=8, value=0)
total_of_special_requests = st.number_input("Total Special Requests", min_value=0, max_value=5, value=0)

# Categorical features with assumed options (adjust based on your images)
hotel = st.selectbox("Hotel Type", ['Resort Hotel', 'City Hotel'])
arrival_date_month = st.selectbox("Arrival Date Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
meal = st.selectbox("Meal Type", ['BB', 'FB', 'HB', 'SC', 'Undefined'])
country = st.selectbox("Country", ['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'Other'])  # Add more countries from your data if needed
market_segment = st.selectbox("Market Segment", ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Groups', 'Complementary', 'Aviation', 'Undefined'])
distribution_channel = st.selectbox("Distribution Channel", ['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'])
is_repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1])
reserved_room_type = st.selectbox("Reserved Room Type", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P'])
assigned_room_type = st.selectbox("Assigned Room Type", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'P'])
deposit_type = st.selectbox("Deposit Type", ['No Deposit', 'Non Refund', 'Refundable'])
customer_type = st.selectbox("Customer Type", ['Transient', 'Contract', 'Transient-Party', 'Group'])

# Agent and Company are often numerical/categorical with many values; treating as numerical for simplicity
agent = st.number_input("Agent ID", min_value=0, max_value=535, value=0)  # 0 often means null
company = st.number_input("Company ID", min_value=0, max_value=543, value=0)  # 0 often means null

# Collect all inputs into a dictionary
input_data = {
    'hotel': hotel,
    'lead_time': lead_time,
    'arrival_date_year': arrival_date_year,
    'arrival_date_month': arrival_date_month,
    'arrival_date_week_number': arrival_date_week_number,
    'arrival_date_day_of_month': arrival_date_day_of_month,
    'stays_in_weekend_nights': stays_in_weekend_nights,
    'stays_in_week_nights': stays_in_week_nights,
    'adults': adults,
    'children': children,
    'babies': babies,
    'meal': meal,
    'country': country,
    'market_segment': market_segment,
    'distribution_channel': distribution_channel,
    'is_repeated_guest': is_repeated_guest,
    'previous_cancellations': previous_cancellations,
    'previous_bookings_not_canceled': previous_bookings_not_canceled,
    'reserved_room_type': reserved_room_type,
    'assigned_room_type': assigned_room_type,
    'booking_changes': booking_changes,
    'deposit_type': deposit_type,
    'agent': agent,
    'company': company,
    'days_in_waiting_list': days_in_waiting_list,
    'customer_type': customer_type,
    'adr': adr,
    'required_car_parking_spaces': required_car_parking_spaces,
    'total_of_special_requests': total_of_special_requests
}

# Button to make prediction
if st.button("Predict"):
    # Convert to DataFrame (model expects a DF with one row)
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    try:
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df) if hasattr(model, 'predict_proba') else None
        
        # Display result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.write("**The booking is predicted to be CANCELED.**")
        else:
            st.write("**The booking is predicted to NOT be canceled.**")
        
        if prob is not None:
            st.write(f"Probability of Cancellation: {prob[0][1]:.2%}")
    except Exception as e:

        st.error(f"Error during prediction: {str(e)}. Ensure the input matches the model's expected features.")

