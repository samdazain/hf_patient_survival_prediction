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

# ---------------------------------------------------------------------------
# Configuration and data
# ---------------------------------------------------------------------------

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
    df = pd.read_csv(DATA_PATH)
    return df


CONFIG = load_config()
FEATURES = load_features()
TRAIN_DF = load_training_data()
TARGET = CONFIG["target"]

# ---------------------------------------------------------------------------
# TabPFN client
# ---------------------------------------------------------------------------

def configure_tabpfn_token():
    """Configure TabPFN authentication with a writable cache path.

    tabpfn-client stores the authenticated token in a file under the installed
    package by default. Hugging Face Spaces may mount site-packages as
    read-only, so the token cache is redirected to /tmp.
    """
    token = (
        os.getenv("TABPFN_TOKEN")
        or os.getenv("PRIORLABS_API_KEY")
        or os.getenv("TABPFN_API_KEY")
    )

    if not token:
        raise RuntimeError(
            "TABPFN_TOKEN belum ditemukan. Tambahkan TABPFN_TOKEN "
            "sebagai Secret pada Hugging Face Space."
        )

    import tabpfn_client
    from tabpfn_client.service_wrapper import UserAuthenticationClient

    # tabpfn-client currently defaults its token cache to
    # <site-packages>/tabpfn_client/.tabpfn, which can be read-only on
    # Hugging Face Spaces. Redirect only the token cache to a writable path.
    writable_cache = Path(
        os.getenv("TABPFN_CLIENT_CACHE_DIR", "/tmp/tabpfn_client")
    )
    writable_cache.mkdir(parents=True, exist_ok=True)
    UserAuthenticationClient.CACHED_TOKEN_FILE = writable_cache / "config"

    # Authorize the current process and persist the token to the writable path.
    tabpfn_client.set_access_token(token)


@st.cache_resource(show_spinner=False)
def build_fitted_model():
    """
    Recreate the deployment classifier from model_config.json and fit it on
    the complete deployment training set.

    The project is pinned to TabPFN V2. The current tabpfn-client exposes
    V2 through create_default_for_version(). The config's n_estimators and
    class-probability balancing are applied here.

    Note:
    fit_mode='fit_preprocessors' exists in the project configuration, but
    current tabpfn-client exposes the hosted estimator configuration rather
    than the local OSS fit_mode parameter. Therefore it is not passed as an
    unsupported constructor argument.
    """
    configure_tabpfn_token()

    from tabpfn_client import TabPFNClassifier
    from tabpfn_client.api_models import ModelVersion

    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        n_estimators=int(CONFIG.get("n_estimators", 4)),
        balance_probabilities=bool(
            CONFIG.get("internal_class_balancing", False)
        ),
        # The deployment configuration indicates that softmax temperature
        # adjustment is disabled. 1.0 therefore means no temperature effect.
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


def numeric_default(feature):
    value = float(TRAIN_DF[feature].median())
    return value


def number_input_for(feature, key):
    series = TRAIN_DF[feature]
    minimum = float(series.min())
    maximum = float(series.max())
    default = numeric_default(feature)

    # Keep inputs within the empirical training-data range to avoid silently
    # extrapolating far outside the deployment data distribution.
    if feature in {"age", "ejection_fraction", "serum_sodium"}:
        step = 1.0
        fmt = "%.0f"
    elif feature == "serum_creatinine":
        step = 0.1
        fmt = "%.1f"
    elif feature == "platelets":
        step = 1000.0
        fmt = "%.0f"
    else:
        step = 1.0
        fmt = "%.0f"

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
    selected = st.selectbox(
        FEATURE_LABELS[feature],
        options=options,
        key=key,
    )
    return BINARY_OPTIONS[feature][selected]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: #ffffff;
    }

    .main-title {
        text-align: center;
        font-size: 2.05rem;
        line-height: 1.18;
        font-weight: 800;
        margin: 0.2rem 0 2rem 0;
        color: #111111;
    }

    .section-title {
        text-align: center;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.8rem 0 0.8rem 0;
        color: #111111;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-weight: 600;
        color: #111111;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #eeeeee;
        border-radius: 10px;
        border: 1px solid #d5d5d5;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 10px;
        border: 2px solid #d2d2d2;
        background: #ffffff;
        color: #111111;
        font-weight: 700;
        min-height: 48px;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: #9b9b9b;
        background: #f7f7f7;
        color: #111111;
    }

    .result-card {
        border: 1px solid #8f8f8f;
        border-radius: 18px;
        background: #e9e9e9;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 3px 8px rgba(0,0,0,0.10);
        margin: 0.8rem auto 1.5rem auto;
        max-width: 700px;
    }

    .result-label {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111111;
    }

    .survival {
        color: #00c853;
        font-weight: 800;
    }

    .death {
        color: #d32f2f;
        font-weight: 800;
    }

    .probability-label {
        margin-top: 0.8rem;
        font-size: 1rem;
        font-weight: 600;
    }

    .small-note {
        text-align: center;
        color: #666666;
        font-size: 0.84rem;
        margin-top: 0.2rem;
    }

    .disclaimer {
        margin-top: 1.5rem;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: #f6f6f6;
        color: #555555;
        font-size: 0.8rem;
        text-align: center;
    }

    @media (max-width: 900px) {
        .main-title {
            font-size: 1.45rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">Prediksi Kelangsungan Hidup Pasien Gagal Jantung<br>'
    'Model TabPFN V2</div>',
    unsafe_allow_html=True,
)

# The three-column layout follows the supplied wireframe.
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

st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)

_, button_col, _ = st.columns([1.2, 1.0, 1.2])
with button_col:
    predict_clicked = st.button("Mulai Prediksi", type="primary")

if predict_clicked:
    patient = pd.DataFrame(
        [[
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
        ]],
        columns=FEATURES,
    )

    try:
        with st.spinner("Memuat dan menjalankan model TabPFN V2..."):
            model = build_fitted_model()
            probabilities = np.asarray(model.predict_proba(patient))[0]
            prediction = np.asarray(model.predict(patient))[0]

        # The training data uses DEATH_EVENT=0 for survival and =1 for death.
        prob_by_class = {
            int(cls): float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }
        survival_prob = prob_by_class.get(0, 0.0)
        death_prob = prob_by_class.get(1, 0.0)

        prediction_text = "Meninggal" if int(prediction) == 1 else "Hidup"
        prediction_class = "death" if int(prediction) == 1 else "survival"

        st.markdown(
            '<div class="section-title">Hasil Prediksi</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Kelangsungan Hidup Pasien Terdeteksi:
                </div>
                <div style="font-size:1.45rem; margin-top:0.35rem;">
                    <span class="{prediction_class}">{prediction_text}</span>
                </div>
                <div class="probability-label">Dengan Probabilitas Model:</div>
                <div>
                    Probabilitas Hidup:
                    <span class="survival"><b>{survival_prob:.0%}</b></span>
                </div>
                <div>
                    Probabilitas Meninggal:
                    <span class="death"><b>{death_prob:.0%}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state["last_prediction"] = {
            "prediction": prediction_text,
            "survival_prob": survival_prob,
            "death_prob": death_prob,
        }

    except Exception as exc:
        st.error(
            "Prediksi gagal dijalankan. Pastikan Secret token TabPFN sudah "
            "dikonfigurasi pada Hugging Face Space dan dependency berhasil "
            "terpasang."
        )
        st.exception(exc)


st.markdown(
    """
    <div class="disclaimer">
        Aplikasi ini merupakan prototipe penelitian untuk prediksi berbasis
        data dan bukan pengganti diagnosis, keputusan klinis, atau konsultasi
        tenaga kesehatan.
    </div>
    """,
    unsafe_allow_html=True,
)
