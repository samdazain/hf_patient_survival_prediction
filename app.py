"""
Prediksi Kelangsungan Hidup Pasien Gagal Jantung
TabPFN V2 (paket lokal `tabpfn`)

Dua halaman:
  1. Fitur terpilih (3 fitur)  -> halaman utama
  2. Seluruh fitur (11 fitur)  -> sub menu di sidebar

Pengaturan model diambil dari notebook HF_TabPFNV2_SHAP_TopK_Sequential.ipynb
(lihat blok KONFIGURASI). SHAP hanya dipakai di notebook untuk memilih fitur.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "deployment_training_data.csv"

st.set_page_config(
    page_title="Prediksi Kelangsungan Hidup Pasien Gagal Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# KONFIGURASI (bersumber dari notebook)
# ---------------------------------------------------------------------------

TARGET = "DEATH_EVENT"          # 0 = hidup, 1 = meninggal
RANDOM_SEED = 42

# Urutan fitur sama dengan FEATURE_NAMES di notebook.
ALL_FEATURES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
]

MODES = {
    # Jalur "TabPFN V2 + SHAP" -> konfigurasi final: 5_Fold, ICB=ON, n_estimators=1,
    # K terbaik = Top-3 (ejection_fraction, serum_creatinine, age).
    "selected": {
        "title": "Fitur Terpilih dengan Model TabPFN V2 + SHAP",
        "features": ["ejection_fraction", "serum_creatinine", "age"],
        "n_estimators": 1,
        "balance_probabilities": True,
    },
    # Jalur "TabPFN V2" (tanpa SHAP, seluruh fitur) -> konfigurasi final:
    # 3_Fold, ICB=ON, n_estimators=8.
    "full": {
        "title": "Seluruh Fitur dengan Model TabPFN V2",
        "features": ALL_FEATURES,
        "n_estimators": 8,
        "balance_probabilities": True,
    },
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data
def load_training_data():
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan pada dataset: {missing}")
    return df


TRAIN_DF = load_training_data()

# ---------------------------------------------------------------------------
# Model (TabPFN V2 lokal)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_model(mode_key):
    """
    Buat TabPFN V2 sesuai notebook, lalu fit pada seluruh data deployment.

    Notebook: TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, n_estimators=..., balance_probabilities=...)
    `fit_mode` tidak diubah (default paket = 'fit_preprocessors',
    sama dengan nilai lama pada notebook).
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    cfg = MODES[mode_key]
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        n_estimators=int(cfg["n_estimators"]),
        balance_probabilities=bool(cfg["balance_probabilities"]),
    )
    X_train = TRAIN_DF[cfg["features"]].to_numpy(dtype=float)
    y_train = TRAIN_DF[TARGET].astype(int).to_numpy()
    model.fit(X_train, y_train)
    return model


def run_prediction(mode_key, values):
    model = get_model(mode_key)
    x = np.asarray([values], dtype=float)
    proba = np.asarray(model.predict_proba(x))[0]
    pred = int(np.asarray(model.predict(x))[0])
    by_class = {int(c): float(p) for c, p in zip(model.classes_, proba)}
    return {
        "prediction": pred,
        "survival_prob": by_class.get(0, 0.0),
        "death_prob": by_class.get(1, 0.0),
    }


# ---------------------------------------------------------------------------
# Label dan input
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
    minimum, maximum = float(series.min()), float(series.max())
    default = float(series.median())

    # Input dibatasi pada rentang data training agar model tidak mengekstrapolasi.
    if feature == "serum_creatinine":
        step, fmt = 0.1, "%.1f"
    elif feature == "platelets":
        step, fmt = 1000.0, "%.0f"
    else:
        step, fmt = 1.0, "%.0f"

    return st.number_input(
        FEATURE_LABELS[feature],
        min_value=minimum,
        max_value=maximum,
        value=min(max(default, minimum), maximum),
        step=step,
        format=fmt,
        key=key,
    )


def binary_input_for(feature, key):
    selected = st.selectbox(
        FEATURE_LABELS[feature],
        options=list(BINARY_OPTIONS[feature].keys()),
        key=key,
    )
    return BINARY_OPTIONS[feature][selected]


def input_for(mode_key, feature):
    key = f"{mode_key}_{feature}"
    if feature in BINARY_OPTIONS:
        return binary_input_for(feature, key)
    return number_input_for(feature, key)


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
        color: #111111;
    }

    .result-label {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111111;
    }

    .survival { color: #00c853; font-weight: 800; }
    .death { color: #d32f2f; font-weight: 800; }

    .probability-label {
        margin-top: 0.8rem;
        font-size: 1rem;
        font-weight: 600;
        color: #111111;
    }

    @media (max-width: 900px) {
        .main-title { font-size: 1.45rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Komponen halaman
# ---------------------------------------------------------------------------


def render_inputs(mode_key):
    """Tata letak 3 kolom. Mengembalikan nilai sesuai urutan MODES[mode_key]['features']."""
    feats = MODES[mode_key]["features"]
    collected = {}

    if mode_key == "selected":
        cols = st.columns(len(feats), gap="large")
        for col, feat in zip(cols, feats):
            with col:
                collected[feat] = input_for(mode_key, feat)
    else:
        left, middle, right = st.columns(3, gap="large")
        layout = [
            (left, ["age", "anaemia", "creatinine_phosphokinase", "diabetes"]),
            (middle, ["ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine"]),
            (right, ["serum_sodium", "sex", "smoking"]),
        ]
        for col, group in layout:
            with col:
                for feat in group:
                    collected[feat] = input_for(mode_key, feat)

    return tuple(float(collected[f]) for f in feats)


def render_result(result):
    pred = result["prediction"]
    text = "Meninggal" if pred == 1 else "Hidup"
    css = "death" if pred == 1 else "survival"

    st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Kelangsungan Hidup Pasien Terdeteksi:</div>
            <div style="font-size:1.45rem; margin-top:0.35rem;">
                <span class="{css}">{text}</span>
            </div>
            <div class="probability-label">Dengan Probabilitas Model:</div>
            <div>Probabilitas Hidup:
                <span class="survival"><b>{result['survival_prob']:.1%}</b></span></div>
            <div>Probabilitas Meninggal:
                <span class="death"><b>{result['death_prob']:.1%}</b></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page(mode_key):
    cfg = MODES[mode_key]
    st.markdown(
        '<div class="main-title">Prediksi Kelangsungan Hidup Pasien Gagal Jantung<br>'
        f'{cfg["title"]}</div>',
        unsafe_allow_html=True,
    )

    # Siapkan model saat halaman dibuka agar tombol prediksi langsung responsif.
    try:
        with st.spinner("Menyiapkan model TabPFN V2..."):
            get_model(mode_key)
    except Exception as exc:
        st.error(
            "Model gagal dimuat. Pastikan paket `tabpfn` dan `torch` terpasang, "
            "serta server dapat mengunduh bobot TabPFN V2 dari Hugging Face."
        )
        st.exception(exc)
        st.stop()

    values = render_inputs(mode_key)

    st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
    _, button_col, _ = st.columns([1.2, 1.0, 1.2])
    with button_col:
        clicked = st.button("Mulai Prediksi", type="primary", key=f"{mode_key}_predict")

    state_key = f"result_{mode_key}"

    if clicked:
        try:
            with st.spinner("Menjalankan prediksi..."):
                result = run_prediction(mode_key, values)
            result["inputs"] = values
            st.session_state[state_key] = result
        except Exception as exc:
            st.session_state.pop(state_key, None)
            st.error("Prediksi gagal dijalankan.")
            st.exception(exc)

    saved = st.session_state.get(state_key)
    if saved is not None:
        if tuple(saved["inputs"]) != tuple(values):
            st.info("Input telah berubah. Tekan “Mulai Prediksi” untuk memperbarui hasil.")
        render_result(saved)

    st.markdown(
        """
        <div style="background-color: #fff3cd; border-left: 5px solid #ff6b6b; border-radius: 5px; padding: 20px; margin-top: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #d32f2f; margin-top: 0;">⚠️ DISCLAIMER PENTING</h3>
            <p style="color: #333; font-weight: 500; line-height: 1.6;">
                Aplikasi ini merupakan <strong>prototipe penelitian</strong> untuk prediksi berbasis
                data dan <strong>BUKAN pengganti</strong> diagnosis, keputusan klinis, atau konsultasi
                tenaga kesehatan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_selected():
    render_page("selected")


def page_full():
    render_page("full")


# ---------------------------------------------------------------------------
# Navigasi: halaman utama = 3 fitur terpilih, sub menu = 11 fitur
# ---------------------------------------------------------------------------

pages = [
    st.Page(
        page_selected,
        title="Prediksi 3 Fitur Terpilih",
        icon="🎯",
        url_path="fitur-terpilih",
        default=True,
    ),
    st.Page(
        page_full,
        title="Prediksi Seluruh Fitur (11)",
        icon="🧬",
        url_path="seluruh-fitur",
    ),
]
st.navigation(pages).run()
