import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "model_config.json"
FEATURES_PATH = APP_DIR / "selected_features.json"
DATA_PATH = APP_DIR / "deployment_training_data.csv"

st.set_page_config(
    page_title="Prediksi Kelangsungan Hidup Pasien Gagal Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_features():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_training_data():
    return pd.read_csv(DATA_PATH)


CONFIG = load_config()
FEATURES = load_features()
TRAIN_DF = load_training_data()
TARGET = CONFIG["target"]


# ---------------------------------------------------------------------------
# TabPFN model
# ---------------------------------------------------------------------------

def configure_tabpfn_token():
    """Use a Hugging Face Space Secret if available."""
    token = (
        os.getenv("TABPFN_TOKEN")
        or os.getenv("PRIORLABS_API_KEY")
        or os.getenv("TABPFN_API_KEY")
    )
    if token:
        try:
            import tabpfn_client
            tabpfn_client.set_access_token(token)
        except Exception:
            pass


@st.cache_resource(show_spinner=False)
def build_fitted_model():
    """Create and fit the configured TabPFN V2 model on the full deployment data."""
    configure_tabpfn_token()

    from tabpfn_client import TabPFNClassifier
    from tabpfn_client.api_models import ModelVersion

    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        n_estimators=int(CONFIG.get("n_estimators", 4)),
        balance_probabilities=bool(
            CONFIG.get("internal_class_balancing", False)
        ),
        softmax_temperature=1.0,
        average_before_softmax=bool(
            CONFIG.get("average_before_softmax", False)
        ),
    )

    X_train = TRAIN_DF[FEATURES].copy()
    y_train = TRAIN_DF[TARGET].copy()
    model.fit(
        X_train,
        y_train,
        description="Heart Failure Survival Prediction - deployment training set",
    )
    return model


# ---------------------------------------------------------------------------
# Dynamic local SHAP
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_shap_background():
    """
    Build one representative reference patient from the deployment data.

    Median is used for continuous variables and mode for binary variables.
    This keeps the model-agnostic local SHAP calculation lightweight enough
    for an interactive web application while still using the actual fitted
    TabPFN model for every prediction queried by SHAP.
    """
    row = {}
    binary_features = {
        "anaemia",
        "diabetes",
        "high_blood_pressure",
        "sex",
        "smoking",
    }

    for feature in FEATURES:
        series = TRAIN_DF[feature]
        if feature in binary_features:
            row[feature] = int(series.mode(dropna=True).iloc[0])
        else:
            row[feature] = float(series.median())

    return pd.DataFrame([row], columns=FEATURES)


@st.cache_data(show_spinner=False)
def calculate_local_shap(patient_values):
    """Calculate patient-specific SHAP contributions for survival probability."""
    import shap

    model = build_fitted_model()
    patient = pd.DataFrame([patient_values], columns=FEATURES)
    background = get_shap_background()

    def predict_survival_probability(x):
        x_df = pd.DataFrame(x, columns=FEATURES)
        proba = np.asarray(model.predict_proba(x_df))
        classes = [int(c) for c in model.classes_]
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(x_df), dtype=float)

    # TabPFN is a black-box/hosted predictor, so a model-agnostic SHAP
    # explainer is appropriate. Five permutations provide a practical
    # latency/quality trade-off for an interactive deployment.
    explainer = shap.PermutationExplainer(
        predict_survival_probability,
        background,
        feature_names=FEATURES,
        seed=42,
    )
    explanation = explainer(
        patient,
        max_evals=(2 * len(FEATURES) * 5) + 1,
        silent=True,
    )

    values = np.asarray(explanation.values)
    if values.ndim > 1:
        values = values[0]

    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    return values.tolist(), base_value


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "age": "Usia",
    "anaemia": "Anemia",
    "creatinine_phosphokinase": "Kreatinin Fosfokinase",
    "diabetes": "Diabetes",
    "ejection_fraction": "Fraksi Ejeksi",
    "high_blood_pressure": "Tekanan Darah Tinggi",
    "platelets": "Platelets",
    "serum_creatinine": "Serum Kreatinin",
    "serum_sodium": "Serum Natrium",
    "sex": "Jenis Kelamin",
    "smoking": "Merokok",
}

BINARY_OPTIONS = {
    "anaemia": {"Tidak": 0, "Ya": 1},
    "diabetes": {"Tidak": 0, "Ya": 1},
    "high_blood_pressure": {"Tidak": 0, "Ya": 1},
    "sex": {"Perempuan": 0, "Laki-laki": 1},
    "smoking": {"Tidak": 0, "Ya": 1},
}


def number_input_for(feature, key):
    series = TRAIN_DF[feature]
    minimum = float(series.min())
    maximum = float(series.max())
    default = float(series.median())

    if feature in {"age", "ejection_fraction", "serum_sodium"}:
        step, fmt = 1.0, "%.0f"
    elif feature == "serum_creatinine":
        step, fmt = 0.1, "%.1f"
    elif feature == "platelets":
        step, fmt = 1000.0, "%.0f"
    else:
        step, fmt = 1.0, "%.0f"

    return st.number_input(
        FEATURE_LABELS[feature],
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step,
        format=fmt,
        key=key,
    )


def binary_input_for(feature, key):
    options = list(BINARY_OPTIONS[feature].keys())
    selected = st.selectbox(FEATURE_LABELS[feature], options=options, key=key)
    return BINARY_OPTIONS[feature][selected]


def render_local_shap(values, patient_values, base_value, survival_probability):
    """Render a dynamic local SHAP contribution chart and value table."""
    import matplotlib.pyplot as plt

    labels = [FEATURE_LABELS[f] for f in FEATURES]
    values = np.asarray(values, dtype=float)
    patient_values = np.asarray(patient_values, dtype=float)

    order = np.argsort(np.abs(values))[::-1]
    ordered_labels = [labels[i] for i in order]
    ordered_values = values[order]
    ordered_inputs = patient_values[order]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(ordered_labels[::-1], ordered_values[::-1])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Kontribusi SHAP terhadap probabilitas meninggal")
    ax.set_title("SHAP Lokal — Kontribusi Fitur pada Pasien Ini")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    direction = np.where(
        ordered_values > 0,
        "Meningkatkan probabilitas meninggal",
        "Menurunkan probabilitas meninggal",
    )
    contribution_df = pd.DataFrame(
        {
            "Fitur": ordered_labels,
            "Nilai Input": ordered_inputs,
            "Kontribusi SHAP": ordered_values,
            "Interpretasi": direction,
        }
    )

    with st.expander("Lihat nilai kontribusi SHAP", expanded=True):
        st.dataframe(
            contribution_df.style.format({"Kontribusi SHAP": "{:+.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Base value probabilitas meninggal: {base_value:.3f}. "
            f"Probabilitas meninggal pasien: {survival_probability:.3f}. "
            "Nilai SHAP positif mendorong prediksi menuju kelas meninggal, "
            "sedangkan nilai negatif mendorong prediksi menuju kelas hidup."
        )


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stApp { background: #ffffff; }
    .main-title { text-align:center; font-size:2.05rem; line-height:1.18; font-weight:800; margin:.2rem 0 2rem; color:#111; }
    .section-title { text-align:center; font-size:1.15rem; font-weight:700; margin:1.8rem 0 .8rem; color:#111; }
    div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label { font-weight:600; color:#111; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { background:#eeeeee; border-radius:10px; border:1px solid #d5d5d5; }
    div[data-testid="stButton"] > button { width:100%; border-radius:10px; border:2px solid #d2d2d2; background:#fff; color:#111; font-weight:700; min-height:48px; }
    div[data-testid="stButton"] > button:hover { border-color:#9b9b9b; background:#f7f7f7; color:#111; }
    .result-card { border:1px solid #8f8f8f; border-radius:18px; background:#e9e9e9; padding:1.2rem 1.4rem; text-align:center; box-shadow:0 3px 8px rgba(0,0,0,.10); margin:.8rem auto 1.5rem; max-width:700px; }
    .result-label { font-size:1.15rem; font-weight:700; color:#111; }
    .survival { color:#00c853; font-weight:800; }
    .death { color:#d32f2f; font-weight:800; }
    .probability-label { margin-top:.8rem; font-size:1rem; font-weight:600; }
    .small-note { text-align:center; color:#666; font-size:.84rem; margin-top:.2rem; }
    .disclaimer { margin-top:1.5rem; padding:.8rem 1rem; border-radius:10px; background:#f6f6f6; color:#555; font-size:.8rem; text-align:center; }
    @media (max-width:900px) { .main-title { font-size:1.45rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">Prediksi Kelangsungan Hidup Pasien Gagal Jantung<br>'
    'Model SHAP + TabPFN V2</div>',
    unsafe_allow_html=True,
)

left, middle, right = st.columns(3, gap="large")

with left:
    age = number_input_for("age", "age")
    anaemia = binary_input_for("anaemia", "anaemia")
    cpk = number_input_for("creatinine_phosphokinase", "cpk")
    diabetes = binary_input_for("diabetes", "diabetes")

with middle:
    ejection_fraction = number_input_for("ejection_fraction", "ejection_fraction")
    high_bp = binary_input_for("high_blood_pressure", "high_bp")
    platelets = number_input_for("platelets", "platelets")
    serum_creatinine = number_input_for("serum_creatinine", "serum_creatinine")

with right:
    serum_sodium = number_input_for("serum_sodium", "serum_sodium")
    sex = binary_input_for("sex", "sex")
    smoking = binary_input_for("smoking", "smoking")

_, button_col, _ = st.columns([1.2, 1.0, 1.2])
with button_col:
    predict_clicked = st.button("Mulai Prediksi", type="primary")

if predict_clicked:
    patient_values = [
        age,
        anaemia,
        cpk,
        diabetes,
        ejection_fraction,
        high_bp,
        platelets,
        serum_creatinine,
        serum_sodium,
        sex,
        smoking,
    ]
    patient = pd.DataFrame([patient_values], columns=FEATURES)

    try:
        with st.spinner("Menjalankan model TabPFN V2... "):
            model = build_fitted_model()
            probabilities = np.asarray(model.predict_proba(patient))[0]
            prediction = np.asarray(model.predict(patient))[0]

        prob_by_class = {
            int(cls): float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }
        survival_prob = prob_by_class.get(0, 0.0)
        death_prob = prob_by_class.get(1, 0.0)

        prediction_text = "Meninggal" if int(prediction) == 1 else "Hidup"
        prediction_class = "death" if int(prediction) == 1 else "survival"

        st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Kelangsungan Hidup Pasien Terdeteksi:</div>
                <div style="font-size:1.45rem; margin-top:.35rem;">
                    <span class="{prediction_class}">{prediction_text}</span>
                </div>
                <div class="probability-label">Dengan Probabilitas Model:</div>
                <div>Probabilitas Hidup: <span class="survival"><b>{survival_prob:.0%}</b></span></div>
                <div>Probabilitas Meninggal: <span class="death"><b>{death_prob:.0%}</b></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Penjelasan Prediksi Pasien</div>', unsafe_allow_html=True)
        with st.spinner("Menghitung kontribusi SHAP untuk pasien ini..."):
            shap_values, base_value = calculate_local_shap(tuple(patient_values))

        render_local_shap(
            shap_values,
            patient_values,
            base_value,
            death_prob,
        )

        st.markdown(
            '<div class="small-note">'
            'Grafik di atas bersifat dinamis dan dihitung ulang berdasarkan nilai input pasien. '
            'SHAP yang ditampilkan adalah SHAP lokal untuk satu prediksi, bukan global feature importance.'
            '</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(
            "Prediksi atau perhitungan SHAP gagal dijalankan. Pastikan Secret "
            "TABPFN_TOKEN sudah dikonfigurasi pada Hugging Face Space dan "
            "seluruh dependency berhasil terpasang."
        )
        st.exception(exc)

st.markdown(
    """
    <div class="disclaimer">
        Aplikasi ini merupakan prototipe penelitian untuk prediksi berbasis data dan
        bukan pengganti diagnosis, keputusan klinis, atau konsultasi tenaga kesehatan.
    </div>
    """,
    unsafe_allow_html=True,
)
