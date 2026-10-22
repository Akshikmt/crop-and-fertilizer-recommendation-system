import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Crop & Fertilizer Advisor",
    page_icon="🌱",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CROP_CSV = BASE_DIR / "crop recommendation.csv"
FERT_CSV = BASE_DIR / "fertilizer recommendation.csv"


@st.cache_resource
def load_models():
    """Train the same Decision Tree approach used in the notebooks from the committed CSVs."""
    if not CROP_CSV.exists() or not FERT_CSV.exists():
        raise FileNotFoundError(
            "Dataset CSVs not found. Keep app.py beside "
            "'crop recommendation.csv' and 'fertilizer recommendation.csv'."
        )

    # Crop model: numeric soil and weather features -> crop label
    crop_df = pd.read_csv(CROP_CSV)
    crop_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    crop_X = crop_df[crop_features].copy()
    crop_y = crop_df["label"].astype(str)

    crop_scaler = StandardScaler()
    crop_X_scaled = crop_scaler.fit_transform(crop_X)
    crop_model = DecisionTreeClassifier(random_state=42)
    crop_model.fit(crop_X_scaled, crop_y)

    # Fertilizer model: encode the two text inputs, scale features, predict fertilizer
    fert_df = pd.read_csv(FERT_CSV)
    fert_df.columns = fert_df.columns.str.strip()

    soil_encoder = LabelEncoder()
    crop_type_encoder = LabelEncoder()
    fert_df["Soil Type"] = soil_encoder.fit_transform(fert_df["Soil Type"].astype(str))
    fert_df["Crop Type"] = crop_type_encoder.fit_transform(fert_df["Crop Type"].astype(str))

    fert_features = [
        "Temparature", "Humidity", "Moisture", "Soil Type", "Crop Type",
        "Nitrogen", "Potassium", "Phosphorous"
    ]
    fert_X = fert_df[fert_features].copy()
    fert_y = fert_df["Fertilizer Name"].astype(str)

    fert_scaler = StandardScaler()
    fert_X_scaled = fert_scaler.fit_transform(fert_X)
    fert_model = DecisionTreeClassifier(random_state=42)
    fert_model.fit(fert_X_scaled, fert_y)

    return (
        crop_model, crop_scaler, crop_features,
        fert_model, fert_scaler, soil_encoder, crop_type_encoder, fert_features
    )


try:
    (
        crop_model, crop_scaler, crop_features,
        fert_model, fert_scaler, soil_encoder, crop_type_encoder, fert_features
    ) = load_models()
except Exception as exc:
    st.error(f"Could not load the models or datasets: {exc}")
    st.stop()

st.title("🌱 Crop & Fertilizer Recommendation System")
st.write(
    "Explore crop and fertilizer recommendations using the project's datasets "
    "and Decision Tree models. Enter measured values for your field below."
)
st.caption("Educational decision-support demo. For real farming decisions, verify results with local agricultural experts and soil testing.")

crop_tab, fert_tab = st.tabs(["🌾 Crop Recommendation", "🧪 Fertilizer Recommendation"])

with crop_tab:
    st.subheader("Find a suitable crop")
    st.write("Enter soil nutrient values and environmental conditions.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        k = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=50.0, step=1.0)
    with c2:
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    with c3:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=100.0, step=5.0)

    if st.button("Recommend Crop", type="primary", key="crop_button"):
        crop_input = pd.DataFrame(
            [[n, p, k, temperature, humidity, ph, rainfall]],
            columns=crop_features,
        )
        prediction = crop_model.predict(crop_scaler.transform(crop_input))[0]
        st.success(f"Recommended crop: **{str(prediction).title()}**")
        st.info("This prediction is based on patterns learned from the crop dataset; check local climate, season, and soil-test guidance before planting.")

with fert_tab:
    st.subheader("Find a suitable fertilizer")
    st.write("Enter the conditions and crop/soil types used by the fertilizer dataset.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        temp_f = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=30.0, step=1.0, key="fert_temp")
        humidity_f = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0, key="fert_humidity")
        moisture = st.number_input("Soil Moisture", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
    with fc2:
        soil_type = st.selectbox("Soil Type", options=sorted(soil_encoder.classes_.tolist()))
        crop_type = st.selectbox("Crop Type", options=sorted(crop_type_encoder.classes_.tolist()))
        nitrogen_f = st.number_input("Nitrogen", min_value=0.0, max_value=200.0, value=20.0, step=1.0, key="fert_n")
    with fc3:
        potassium_f = st.number_input("Potassium", min_value=0.0, max_value=200.0, value=10.0, step=1.0, key="fert_k")
        phosphorus_f = st.number_input("Phosphorus", min_value=0.0, max_value=200.0, value=20.0, step=1.0, key="fert_p")

    if st.button("Recommend Fertilizer", type="primary", key="fert_button"):
        try:
            soil_code = soil_encoder.transform([soil_type])[0]
            crop_code = crop_type_encoder.transform([crop_type])[0]
            fert_input = pd.DataFrame(
                [[temp_f, humidity_f, moisture, soil_code, crop_code,
                  nitrogen_f, potassium_f, phosphorus_f]],
                columns=fert_features,
            )
            prediction = fert_model.predict(fert_scaler.transform(fert_input))[0]
            st.success(f"Recommended fertilizer: **{prediction}**")
            st.info("Use the recommendation as a model output, not a guaranteed agronomic prescription. Confirm nutrient needs and application rates with a soil test or local expert.")
        except ValueError as exc:
            st.error(f"Could not process these values: {exc}")

with st.expander("About this app"):
    st.write(
        "The app trains Decision Tree classifiers from the two CSV datasets when it first starts, "
        "then caches the fitted models for subsequent interactions. The notebooks' exported .pkl files "
        "are not currently present in the repository, so this version does not load those pickle files."
    )
