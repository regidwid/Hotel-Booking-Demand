# app.py
# Streamlit UI for a saved sklearn Pipeline (bagging model) that already includes preprocessing.
# Put "hotel_booking_prediction.sav" in the same folder, or point the sidebar to the file.

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import BaggingClassifier


# Joblib first, pickle as fallback
def _load_any(path):
    import joblib, pickle
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

st.set_page_config(page_title="Hotel Booking Predictor", page_icon="🏨", layout="centered")

# Defaults from your screenshots
DEFAULT_SCHEMA = {
    "numerical": [
        "previous_cancellations",
        "booking_changes",
        "days_in_waiting_list",
        "required_car_parking_spaces",
        "total_of_special_requests",
    ],
    "categorical": [
        "country",
        "market_segment",
        "deposit_type",
        "customer_type",
        "reserved_room_type",
    ],
    "options": {
        # Small, friendly fallback list for country; will be replaced if the model reveals full categories
        "country": ["PRT", "GBR", "FRA", "ESP", "DEU", "USA", "IRL", "NLD", "BRA", "BEL", "CHE", "ITA", "NOR", "SWE", "AUS", "CHN", "RUS", "CAN", "POL"],
        "market_segment": ["Offline TA/TO", "Online TA", "Direct", "Groups", "Corporate", "Complementary", "Aviation", "Undefined"],
        "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
        "customer_type": ["Transient-Party", "Transient", "Contract", "Group"],
        "reserved_room_type": ["A", "E", "D", "F", "B", "G", "C", "H", "L", "P"],
    },
}

# Input ranges from your screenshots (kept generous where the screenshot shows many values)
NUMERIC_RANGES = {
    "previous_cancellations": (0, 26, 0),      # min, max, default
    "booking_changes": (0, 21, 0),
    "days_in_waiting_list": (0, 400, 0),
    "required_car_parking_spaces": (0, 2, 0),
    "total_of_special_requests": (0, 5, 0),
}

@st.cache_resource(show_spinner=False)
def load_model(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return _load_any(str(path))

def infer_schema_from_model(model):
    # Try to discover original feature groups and categorical options from the pipeline’s ColumnTransformer
    schema = {"numerical": [], "categorical": [], "options": {}}

    def walk(obj):
        # Look for ColumnTransformer-like objects with transformers_
        if hasattr(obj, "transformers_"):
            for name, transformer, cols in obj.transformers_:
                # Try to find OneHotEncoder in this branch
                ohe = None
                # Direct OHE
                if hasattr(transformer, "categories_"):
                    ohe = transformer
                # Or OHE inside a sub-pipeline
                if hasattr(transformer, "named_steps"):
                    for s in transformer.named_steps.values():
                        if hasattr(s, "categories_"):
                            ohe = s
                if ohe is not None:
                    try:
                        for col, cat in zip(cols, ohe.categories_):
                            col = col if isinstance(col, str) else str(col)
                            if col not in schema["categorical"]:
                                schema["categorical"].append(col)
                            # Clean out NaN-like strings
                            clean = [str(x) for x in cat if str(x).lower() not in {"nan", "none"}]
                            schema["options"][col] = clean
                    except Exception:
                        pass
                else:
                    # Treat as numerical if we didn't identify an encoder
                    if isinstance(cols, (list, tuple, np.ndarray)):
                        for c in cols:
                            c = c if isinstance(c, str) else str(c)
                            if c not in schema["numerical"]:
                                schema["numerical"].append(c)
        # Recurse through nested pipelines
        if hasattr(obj, "named_steps"):
            for s in obj.named_steps.values():
                walk(s)

    walk(model)
    return schema

def merge_schema(inferred, default):
    merged = {
        "numerical": list(default["numerical"]),
        "categorical": list(default["categorical"]),
        "options": {k: list(v) for k, v in default["options"].items()},
    }
    # Add any inferred numerical not in defaults
    for c in inferred.get("numerical", []):
        if c not in merged["numerical"] and c not in merged["categorical"]:
            merged["numerical"].append(c)
    # Add inferred categorical and their options
    for c in inferred.get("categorical", []):
        if c not in merged["categorical"]:
            # Avoid duplicates if already marked as numerical by default
            if c in merged["numerical"]:
                merged["numerical"].remove(c)
            merged["categorical"].append(c)
    for c, opts in inferred.get("options", {}).items():
        merged["options"][c] = list(opts)
    return merged

st.sidebar.title("Model")
model_path = st.sidebar.text_input("Path to model (.sav)", value="hotel_booking_prediction.sav")
load_btn = st.sidebar.button("Load / Reload model", type="primary")

# Lazy-load on first render or when user clicks
if "model_obj" not in st.session_state or load_btn:
    try:
        st.session_state.model_obj = load_model(model_path)
        st.session_state.inferred_schema = infer_schema_from_model(st.session_state.model_obj)
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

model = st.session_state.get("model_obj", None)
schema_inferred = st.session_state.get("inferred_schema", {"numerical": [], "categorical": [], "options": {}})
schema = merge_schema(schema_inferred, DEFAULT_SCHEMA)

st.title("🏨 Hotel Booking Prediction")
st.caption("Enter raw feature values. The pipeline inside the model will handle preprocessing.")

# Helpful peek
with st.expander("What did the app find inside your pipeline?"):
    st.write("Inferred numerical:", schema_inferred.get("numerical", []))
    st.write("Inferred categorical:", schema_inferred.get("categorical", []))
    if schema_inferred.get("options", {}).get("country"):
        st.write(f"Found {len(schema_inferred['options']['country'])} country codes from the model.")

# Randomize/reset helpers
def randomize_inputs():
    for k, (mn, mx, _) in NUMERIC_RANGES.items():
        if k in st.session_state:
            st.session_state[k] = int(np.random.randint(mn, mx + 1))
    # For cats, pick a random option if we have them
    for c in schema["categorical"]:
        opts = schema["options"].get(c, [])
        if opts and c in st.session_state:
            st.session_state[c] = np.random.choice(opts)

left, right = st.columns(2)

# Build the form
with st.form("booking_form", clear_on_submit=False):
    st.subheader("Inputs")

    with left:
        # Numerical controls
        num_vals = {}
        for col in schema["numerical"]:
            if col in NUMERIC_RANGES:
                mn, mx, default = NUMERIC_RANGES[col]
            else:
                # Unknown numeric: give a broad default
                mn, mx, default = 0, 1000, 0
            num_vals[col] = st.number_input(
                col,
                min_value=int(mn),
                max_value=int(mx),
                value=int(default),
                step=1,
                key=col,
            )

    with right:
        # Categorical controls
        cat_vals = {}
        for col in schema["categorical"]:
            options = schema["options"].get(col, [])
            # Clean possible duplicates and maintain order
            options = list(dict.fromkeys(options))
            # If model didn't reveal options (e.g., unknown col), allow free text
            if len(options) == 0:
                cat_vals[col] = st.text_input(col, key=col)
            else:
                # Some lists (like country) can be very long; Streamlit selectbox is searchable
                # Remove NaN-like entries just in case
                options = [o for o in options if str(o).lower() not in {"nan", "none"}]
                default_ix = 0 if len(options) > 0 else None
                cat_vals[col] = st.selectbox(col, options=options, index=default_ix, key=col)

    # Action buttons inside the form
    b1, b2, b3 = st.columns(3)
    submit = b1.form_submit_button("Predict", type="primary")
    rnd = b2.form_submit_button("Randomize")
    rst = b3.form_submit_button("Reset to defaults")

    if rnd:
        randomize_inputs()
        st.experimental_rerun()
    if rst:
        for k, (_, _, default) in NUMERIC_RANGES.items():
            st.session_state[k] = int(default)
        for c in schema["categorical"]:
            if schema["options"].get(c):
                st.session_state[c] = schema["options"][c][0]
        st.experimental_rerun()

# Do prediction
if submit:
    if model is None:
        st.error("Load the model from the sidebar first.")
        st.stop()

    # Collect the current values from session_state to ensure we read the latest
    row = {}
    for col in schema["numerical"]:
        row[col] = int(st.session_state.get(col))
    for col in schema["categorical"]:
        row[col] = st.session_state.get(col)

    X = pd.DataFrame([row])
    st.write("Input row:", X)

    try:
        y_pred = model.predict(X)
        st.success(f"Prediction: {y_pred[0] if len(y_pred) else y_pred}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Show class probabilities if available (typical for classification)
    proba = None
    try:
        proba = model.predict_proba(X)
    except Exception:
        proba = None

    if proba is not None:
        try:
            classes = None
            # Try to grab classes_ from the final estimator inside the pipeline
            final_est = model
            if hasattr(model, "named_steps"):
                # Guess the final estimator as the last step
                last_step_name = list(model.named_steps.keys())[-1]
                final_est = model.named_steps[last_step_name]
            if hasattr(final_est, "classes_"):
                classes = list(final_est.classes_)
            else:
                classes = list(range(proba.shape[1]))
            probs = {str(c): float(p) for c, p in zip(classes, proba[0])}
            st.write("Class probabilities:", probs)
        except Exception:
            pass

# Little “how to”
with st.expander("How to run this app"):
    st.write("- Install: pip install streamlit scikit-learn pandas joblib")
    st.write("- Put hotel_booking_prediction.sav next to app.py (or point to it in the sidebar).")
    st.write("- Run: streamlit run app.py")

with st.expander("Notes / Troubleshooting"):
    st.write("- The app lets the pipeline handle all encoding and scaling; just type the raw values.")
    st.write("- If your saved pipeline exposes OneHotEncoder categories, the app will read them (e.g., full country list).")
    st.write("- If your model needs extra features not in the defaults, the app will try to discover them; if not, add them to DEFAULT_SCHEMA/NUMERIC_RANGES.")
    st.write("- If you get 'unknown category' errors, your encoder may not ignore unknowns. Use one of the known options the model reveals.")


