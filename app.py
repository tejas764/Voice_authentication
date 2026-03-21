import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import parameters as p
from auth_service import VoiceAuthService

st.set_page_config(page_title="Voice Authentication", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg: #f3f4ee;
            --card: #ffffff;
            --ink: #102a43;
            --muted: #486581;
            --accent: #1f7a8c;
            --accent-soft: #dff4f8;
            --success: #2f9e44;
            --danger: #d9480f;
            --border: #d9e2ec;
        }

        .stApp {
            background: radial-gradient(circle at 20% 20%, #dff4f8 0%, transparent 35%),
                        radial-gradient(circle at 80% 10%, #ffe8cc 0%, transparent 35%),
                        var(--bg);
            color: var(--ink);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }

        .hero {
            background: linear-gradient(115deg, #0e7490, #1f7a8c 60%, #2a9d8f);
            border-radius: 20px;
            padding: 28px;
            color: #f8fafc;
            box-shadow: 0 18px 34px rgba(16, 42, 67, 0.22);
            margin-bottom: 18px;
            animation: fadeIn 0.5s ease-out;
        }

        .hero p {
            color: #e6f7fa;
            margin-bottom: 0;
        }

        .metric-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 16px;
            box-shadow: 0 8px 20px rgba(16, 42, 67, 0.08);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_service() -> VoiceAuthService:
    return VoiceAuthService()


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def save_audio_bytes_to_wav(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        return tmp.name


service = get_service()

st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;">Voice Authentication Studio</h1>
        <p style="margin-top:8px;">Enroll voiceprints, run recognition, and manage your speaker database in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

users = service.list_enrolled_users()

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"<div class='metric-card'><b>Enrolled Users</b><br><span style='font-size:1.4rem'>{len(users)}</span></div>", unsafe_allow_html=True)
with col_b:
    st.markdown(f"<div class='metric-card'><b>Metric</b><br><span style='font-size:1.2rem'>{p.COST_METRIC}</span></div>", unsafe_allow_html=True)
with col_c:
    st.markdown(f"<div class='metric-card'><b>Threshold</b><br><span style='font-size:1.2rem'>{p.THRESHOLD}</span></div>", unsafe_allow_html=True)

st.write("")

tab_enroll, tab_recognize, tab_manage = st.tabs(["Enroll", "Recognize", "Manage Users"])

with tab_enroll:
    st.subheader("Single User Enrollment")
    name = st.text_input("Student name")
    register_number = st.text_input("Register number")
    enroll_file = st.file_uploader("Upload a voice sample (.wav/.flac)", type=["wav", "flac"], key="single_enroll")
    live_audio = st.audio_input("Or record live voice sample", key="live_enroll")

    if st.button("Enroll User", type="primary"):
        if not name.strip() or not register_number.strip():
            st.error("Please provide both name and register number.")
        else:
            if enroll_file is None and live_audio is None:
                st.error("Please upload an audio file or record a live sample.")
                st.stop()

            temp_path = save_uploaded_file(enroll_file) if enroll_file is not None else save_audio_bytes_to_wav(live_audio.getvalue())
            try:
                service.enroll_person(name.strip(), register_number.strip(), temp_path)
                st.success(f"Enrolled '{name.strip()}' ({register_number.strip()}) successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Enrollment failed: {exc}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    st.divider()
    st.subheader("Bulk Enrollment (CSV)")
    st.caption("CSV should include columns: filename,speaker")
    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="bulk_enroll")
    if st.button("Enroll from CSV"):
        if csv_file is None:
            st.error("Please upload a CSV file first.")
        else:
            temp_csv = save_uploaded_file(csv_file)
            try:
                total = service.enroll_csv(temp_csv)
                st.success(f"Enrolled {total} users from CSV.")
                st.rerun()
            except Exception as exc:
                st.error(f"Bulk enrollment failed: {exc}")
            finally:
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)

with tab_recognize:
    st.subheader("Recognize Speaker")
    rec_file = st.file_uploader("Upload unknown speaker audio", type=["wav", "flac"], key="recognize")
    rec_live_audio = st.audio_input("Or record live voice sample", key="live_recognize")

    if st.button("Run Recognition", type="primary"):
        if rec_file is None and rec_live_audio is None:
            st.error("Please upload or record an audio file.")
        else:
            temp_path = save_uploaded_file(rec_file) if rec_file is not None else save_audio_bytes_to_wav(rec_live_audio.getvalue())
            try:
                result = service.recognize(temp_path)
                if result["accepted"]:
                    matched = result.get("matched_user")
                    if matched:
                        st.success(f"Recognized: {matched['name']} ({matched['register_number']})")
                    else:
                        st.success(f"Recognized: {result['speaker']}")
                else:
                    st.warning("No confident match found.")

                st.info(
                    f"Score: {result['score']:.4f} | Metric: {result['metric']} | Threshold: {result['threshold']:.4f}"
                )

                distances = pd.DataFrame(
                    [{"speaker": k, "distance": v} for k, v in result["distances"].items()]
                ).sort_values("distance", ascending=True)
                distances["display"] = distances["speaker"]
                users_df = service.list_enrolled_user_details()
                if not users_df.empty:
                    merged = distances.merge(users_df[["speaker_key", "name", "register_number"]], left_on="speaker", right_on="speaker_key", how="left")
                    merged["display"] = merged.apply(
                        lambda row: f"{row['name']} ({row['register_number']})" if pd.notna(row.get("name")) else row["speaker"],
                        axis=1,
                    )
                    distances = merged[["display", "distance"]]
                    distances.columns = ["speaker", "distance"]
                st.dataframe(distances, use_container_width=True)
            except Exception as exc:
                st.error(f"Recognition failed: {exc}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

with tab_manage:
    st.subheader("Enrolled Users")
    users_df = service.list_enrolled_user_details()
    if users_df.empty:
        st.caption("No users enrolled yet.")
    else:
        view = users_df[["name", "register_number", "created_at"]].rename(
            columns={"name": "Name", "register_number": "Register Number", "created_at": "Enrolled At"}
        )
        st.dataframe(view, use_container_width=True)
