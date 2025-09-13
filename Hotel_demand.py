# app.py
import io
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import category_encoders
from sklearn.ensemble import BaggingClassifier

# Try joblib first, then pickle as a fallback
try:
    import joblib
except Exception:
    joblib = None
import pickle

# Optional imports for introspection (won't crash if missing)
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
except Exception:
    ColumnTransformer = None
    OneHotEncoder = None

st.set_page_config(page_title="Hotel Booking Prediction", page_icon="🏨", layout="centered")
warnings.filterwarnings("ignore")

MODEL_FILE = "hotel_booking_prediction_model.sav"

# Expected features (based on your provided screenshots)
EXPECTED_CATEGORICAL = {
    "country": None,  # will be auto-filled from the model if possible
    "market_segment": ["Offline TA/TO", "Online TA", "Direct", "Groups", "Corporate", "Complementary", "Aviation", "Undefined"],
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "customer_type": ["Transient-Party", "Transient", "Contract", "Group"],
    "reserved_room_type": ["A", "E", "D", "F", "B", "G", "C", "H", "L", "P"],
}
# min, max, default
EXPECTED_NUMERIC = {
    "previous_cancellations": (0, 26, 0),
    "booking_changes": (0, 21, 0),
    "days_in_waiting_list": (0, 391, 0),
    "required_car_parking_spaces": (0, 2, 0),
    "total_of_special_requests": (0, 5, 0),
}
EXPECTED_FEATURE_ORDER = list(EXPECTED_CATEGORICAL.keys()) + list(EXPECTED_NUMERIC.keys())


@st.cache_resource(show_spinner=True)
def load_model_from_disk(path: str):
    p = Path(path)
    if not p.exists():
        return None
    # joblib load first
    if joblib is not None:
        try:
            return joblib.load(p)
        except Exception:
            pass
    # pickle fallback
    with open(p, "rb") as f:
        return pickle.load(f)


def load_model_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    data = uploaded_file.read()
    # joblib first
    if joblib is not None:
        try:
            return joblib.load(io.BytesIO(data))
        except Exception:
            pass
    # pickle fallback
    return pickle.loads(data)


def get_final_estimator(model):
    # For pipelines, the last step is the estimator
    try:
        if hasattr(model, "steps") and len(model.steps) > 0:
            return model.steps[-1][1]
    except Exception:
        pass
    return model


def extract_categorical_options(model):
    """
    Try to pull actual category lists (e.g., all countries) from the fitted pipeline.
    Works if there's a ColumnTransformer with OneHotEncoder fitted.
    Returns: dict[col_name] = list_of_categories
    """
    found = {}

    if model is None or ColumnTransformer is None or OneHotEncoder is None:
        return found

    def walk(est, cols_ctx=None):
        try:
            # ColumnTransformer
            if hasattr(est, "transformers_"):
                for name, trans, cols in est.transformers_:
                    # Normalize cols to a list
                    if isinstance(cols, (list, tuple, np.ndarray)):
                        col_list = list(cols)
                    elif isinstance(cols, slice):
                        # can't map slice to names reliably without feature_names_in_
                        col_list = []
                    else:
                        col_list = [cols] if cols is not None else []

                    # If this transformer is itself a pipeline, dive in while keeping the cols
                    if hasattr(trans, "steps"):
                        for _, step in trans.steps:
                            walk(step, col_list)

                    # If it's a fitted OneHotEncoder, pull categories
                    elif isinstance(trans, OneHotEncoder) and hasattr(trans, "categories_"):
                        for i, col in enumerate(col_list):
                            col_name = str(col)
                            try:
                                cats = list(trans.categories_[i])
                                found[col_name] = cats
                            except Exception:
                                pass

            # If the estimator is a OneHotEncoder directly (less common)
            if isinstance(est, OneHotEncoder) and hasattr(est, "categories_") and cols_ctx:
                for i, col in enumerate(cols_ctx):
                    col_name = str(col)
                    try:
                        cats = list(est.categories_[i])
                        found[col_name] = cats
                    except Exception:
                        pass
        except Exception:
            pass

    walk(model)
    # Try to normalize keys to the actual input column names if available
    try:
        if hasattr(model, "feature_names_in_"):
            fni = list(model.feature_names_in_)
            fixed = {}
            for k, v in found.items():
                # If keys are integers (positions), map them to feature names
                if k.isdigit():
                    idx = int(k)
                    if 0 <= idx < len(fni):
                        fixed[fni[idx]] = v
                else:
                    fixed[k] = v
            return fixed
    except Exception:
        pass

    return found


def build_input_ui(cat_options):
    st.subheader("Enter booking details")

    # Country
    country_opts = cat_options.get("country", None)
    if country_opts:
        country = st.selectbox("Country (ISO-3)", country_opts, index=min(country_opts.index("PRT") if "PRT" in country_opts else 0, len(country_opts)-1))
    else:
        country = st.text_input("Country (ISO-3, e.g., PRT, GBR, USA)", value="PRT")

    market_segment = st.selectbox("Market segment", cat_options.get("market_segment", EXPECTED_CATEGORICAL["market_segment"]))
    deposit_type = st.selectbox("Deposit type", cat_options.get("deposit_type", EXPECTED_CATEGORICAL["deposit_type"]))
    customer_type = st.selectbox("Customer type", cat_options.get("customer_type", EXPECTED_CATEGORICAL["customer_type"]))
    reserved_room_type = st.selectbox("Reserved room type", cat_options.get("reserved_room_type", EXPECTED_CATEGORICAL["reserved_room_type"]))

    prev_canc_min, prev_canc_max, prev_canc_def = EXPECTED_NUMERIC["previous_cancellations"]
    previous_cancellations = st.number_input("Previous cancellations", min_value=prev_canc_min, max_value=prev_canc_max, value=prev_canc_def, step=1)

    bc_min, bc_max, bc_def = EXPECTED_NUMERIC["booking_changes"]
    booking_changes = st.number_input("Booking changes", min_value=bc_min, max_value=bc_max, value=bc_def, step=1)

    diw_min, diw_max, diw_def = EXPECTED_NUMERIC["days_in_waiting_list"]
    days_in_waiting_list = st.number_input("Days in waiting list", min_value=diw_min, max_value=diw_max, value=diw_def, step=1)

    park_min, park_max, park_def = EXPECTED_NUMERIC["required_car_parking_spaces"]
    required_car_parking_spaces = st.number_input("Required car parking spaces", min_value=park_min, max_value=park_max, value=park_def, step=1)

    sr_min, sr_max, sr_def = EXPECTED_NUMERIC["total_of_special_requests"]
    total_of_special_requests = st.number_input("Total special requests", min_value=sr_min, max_value=sr_max, value=sr_def, step=1)

    inputs = {
        "country": country,
        "market_segment": market_segment,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "reserved_room_type": reserved_room_type,
        "previous_cancellations": int(previous_cancellations),
        "booking_changes": int(booking_changes),
        "days_in_waiting_list": int(days_in_waiting_list),
        "required_car_parking_spaces": int(required_car_parking_spaces),
        "total_of_special_requests": int(total_of_special_requests),
    }
    return inputs


def make_prediction(model, features: dict):
    df = pd.DataFrame([features])

    # Reorder columns to what the pipeline expects, if available
    try:
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            df = df.reindex(columns=cols)
        else:
            # fallback to our expected order
            df = df.reindex(columns=EXPECTED_FEATURE_ORDER)
    except Exception:
        pass

    pred = model.predict(df)[0]
    proba_map = None

    # Try to get probabilities + class labels from final estimator
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            final_est = get_final_estimator(model)
            classes = getattr(final_est, "classes_", None)
            if classes is None:
                # fallback: label classes as indices
                classes = list(range(len(probs)))
            proba_map = {str(classes[i]): float(probs[i]) for i in range(len(probs))}
    except Exception:
        pass

    return pred, proba_map


def main():
    st.title("Hotel Booking Prediction")
    st.caption("Loads your saved Bagging pipeline with preprocessing and predicts the target for a single booking.")

    # Load model
    model = load_model_from_disk(MODEL_FILE)
    if model is None:
        st.warning(f"Couldn't find {MODEL_FILE}. Upload the .sav file in the sidebar.")
        uploaded = st.sidebar.file_uploader("Upload model (.sav)", type=["sav", "pkl", "pickle"])
        if uploaded:
            model = load_model_from_upload(uploaded)
            if model:
                st.success("Model loaded from upload.")
    else:
        st.success("Model loaded.")

    if model is None:
        st.stop()

    # Try to discover categories from the fitted pipeline
    discovered_cats = extract_categorical_options(model)

    # Merge discovered categories with expected defaults
    merged_cats = {}
    for k in EXPECTED_CATEGORICAL.keys():
        if k in discovered_cats and discovered_cats[k]:
            merged_cats[k] = discovered_cats[k]
        else:
            merged_cats[k] = EXPECTED_CATEGORICAL[k]

    with st.form("predict_form"):
        features = build_input_ui(merged_cats)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            pred, proba_map = make_prediction(model, features)
            st.subheader("Result")
            st.write(f"Predicted class: {pred}")

            if proba_map is not None:
                st.write("Class probabilities:")
                # Show sorted by class label
                for cls, p in sorted(proba_map.items(), key=lambda x: x[0]):
                    st.write(f"  {cls}: {p:.4f}")
                st.caption("Note: If your target is 'is_canceled', probability for class '1' is the cancellation chance.")

            with st.expander("Show input payload"):
                st.json(features)
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

    with st.sidebar:
        st.markdown("About")
        st.write(
            "This app assumes your pipeline handles all preprocessing (encoding/scaling). "
            "It will auto-read allowed categorical values from the fitted pipeline when possible."
        )


if __name__ == "__main__":
    main()