import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once and cache it.
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained model from the specified path."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# Load the pipeline model
model = load_model('hotel_booking_prediction_model.sav')

# --- UI Elements ---
st.title("🏨 Hotel Booking Cancellation Predictor")
st.write(
    "This app predicts whether a hotel booking will be canceled or not. "
    "Adjust the parameters in the sidebar to match the booking details and click 'Predict'."
)
st.markdown("---")


# --- Sidebar for User Input ---
st.sidebar.header("Guest Booking Details")

# Helper function to create inputs
def user_input_features():
    # Based on the provided images for categorical features
    country_list = [
        'PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'OMN', 'ARG', 'POL', 'DEU', 'BEL', 'CHE', 'CN', 'GRC',
        'ITA', 'NLD', 'DNK', 'RUS', 'SWE', 'AUS', 'EST', 'CZE', 'BRA', 'FIN', 'MOZ', 'BWA', 'LUX', 'SVN', 'ALB',
        'IND', 'CHN', 'KOR', 'MEX', 'ISR', 'SRB', 'TWN', 'MAR', 'HRV', 'AUT', 'TZA', 'CYP', 'ZAF', 'LVA', 'JOR',
        'LBN', 'VNM', 'HUN', 'MAC' # Abridged list
    ]
    market_segment = st.sidebar.selectbox('Market Segment',
        ('Online TA', 'Offline TA/TO', 'Direct', 'Groups', 'Corporate', 'Complementary', 'Aviation', 'Undefined'))

    deposit_type = st.sidebar.selectbox('Deposit Type',
        ('No Deposit', 'Non Refund', 'Refundable'))

    customer_type = st.sidebar.selectbox('Customer Type',
        ('Transient', 'Transient-Party', 'Contract', 'Group'))

    reserved_room_type = st.sidebar.selectbox('Reserved Room Type',
        ('A', 'D', 'E', 'F', 'B', 'G', 'C', 'H', 'L', 'P'))

    country = st.sidebar.selectbox('Country', country_list, index=0)

    st.sidebar.markdown("---")

    # Based on the provided images for numerical features
    previous_cancellations = st.sidebar.number_input('Previous Cancellations', min_value=0, max_value=30, value=0, step=1)
    booking_changes = st.sidebar.number_input('Booking Changes', min_value=0, max_value=25, value=0, step=1)
    days_in_waiting_list = st.sidebar.number_input('Days in Waiting List', min_value=0, max_value=400, value=0, step=1)
    required_car_parking_spaces = st.sidebar.slider('Required Car Parking Spaces', 0, 8, 0)
    total_of_special_requests = st.sidebar.slider('Total Special Requests', 0, 5, 1)

    # Create a dictionary of the inputs
    data = {
        'country': country,
        'market_segment': market_segment,
        'deposit_type': deposit_type,
        'customer_type': customer_type,
        'reserved_room_type': reserved_room_type,
        'previous_cancellations': previous_cancellations,
        'booking_changes': booking_changes,
        'days_in_waiting_list': days_in_waiting_list,
        'required_car_parking_spaces': required_car_parking_spaces,
        'total_of_special_requests': total_of_special_requests
    }

    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()


# --- Main Panel for Displaying Input and Prediction ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Booking Details Entered:")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}))

with col2:
    st.subheader("Prediction")
    if st.button('Predict', type="primary", use_container_width=True):
        if model:
            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # Display results
            if prediction[0] == 1:
                st.error("Prediction: Booking WILL be Canceled", icon="❌")
                probability = prediction_proba[0][1] * 100
            else:
                st.success("Prediction: Booking WILL NOT be Canceled", icon="✅")
                probability = prediction_proba[0][0] * 100

            st.metric(label="Confidence", value=f"{probability:.2f}%")

            # Display probability breakdown
            st.write("Prediction Probabilities:")
            proba_df = pd.DataFrame(
                {
                    'Status': ['Not Canceled', 'Canceled'],
                    'Probability': [f"{p*100:.2f}%" for p in prediction_proba[0]]
                }
            )
            st.table(proba_df)
        else:
            st.error("Model is not loaded. Please check the file path and logs.")

st.markdown("---")
st.write("Built by an AI assistant. Ensure the model file `hotel_booking_prediction_model.sav` is in the same directory.")
