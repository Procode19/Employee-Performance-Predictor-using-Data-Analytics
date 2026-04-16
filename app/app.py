import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Employee Intelligence System | Enterprise AI Dashboard",
    layout="wide",
    page_icon="📊"
)

# ================= CUSTOM STYLE =================
st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { color: #00adb5; }
.stMetric { background-color: #111827; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("data/employee_features.csv")

try:
    df = load_data()
except:
    st.error("❌ Dataset not found. Run main.py first to generate data.")
    st.stop()

# ================= HEADER =================
st.title("📊 Employee Intelligence System")
st.markdown("### 🚀 Enterprise AI Dashboard for Workforce Analytics & Performance Prediction")

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("🎛️ Control Panel")

dept_filter = st.sidebar.multiselect(
    "Department",
    df["department"].unique(),
    default=df["department"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    df["gender"].unique(),
    default=df["gender"].unique()
)

filtered_df = df[
    (df["department"].isin(dept_filter)) &
    (df["gender"].isin(gender_filter))
]

# ================= KPI DASHBOARD =================
st.subheader("📌 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Employees", len(filtered_df))
c2.metric("Avg Age", round(filtered_df["age"].mean(), 1))
c3.metric("High Performers",
          len(filtered_df[filtered_df["perf_band_next"] == "High"]))
c4.metric("Departments", filtered_df["department"].nunique())

st.markdown("---")

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(
    ["📊 Workforce Overview", "📈 Analytics Engine", "🤖 AI Prediction Engine"]
)

# ================= TAB 1 =================
with tab1:
    st.subheader("📊 Workforce Overview")

    st.dataframe(filtered_df.head())

    fig, ax = plt.subplots()
    filtered_df["department"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Employees per Department")
    st.pyplot(fig)

# ================= TAB 2 =================
with tab2:
    st.subheader("📈 Analytics Engine")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x="perf_band_next", ax=ax)
        ax.set_title("Performance Distribution")
        st.pyplot(fig)

    with col2:
        st.write("Correlation Heatmap")

        numeric_df = filtered_df.select_dtypes(include=np.number)

        if numeric_df.shape[1] > 1:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)
        else:
            st.warning("Not enough numeric features for correlation.")

# ================= TAB 3 =================
with tab3:
    st.subheader("🤖 AI-Based Performance Prediction")

    st.write("Enter employee attributes:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)

    with col2:
        experience = st.slider("Experience (Years)", 0, 40, 5)

    with col3:
        training = st.slider("Training Hours", 0, 200, 50)

    if st.button("🚀 Predict Performance"):

        # Simple weighted scoring model (industrial-style logic)
        score = (
            (age * 0.15) +
            (experience * 0.55) +
            (training * 0.30)
        )

        if score > 45:
            result = "🟢 High Performer"
        elif score > 25:
            result = "🟡 Medium Performer"
        else:
            result = "🔴 Low Performer"

        st.success(f"Prediction Result: {result}")
        st.info(f"Computed Performance Score: {round(score,2)}")

# ================= DOWNLOAD =================
st.download_button(
    "📥 Download Filtered Report",
    data=filtered_df.to_csv(index=False).encode(),
    file_name="employee_intelligence_report.csv",
    mime="text/csv"
)

# ================= FOOTER =================
st.markdown("---")
st.markdown("### 🚀 Built by Om Navgire | Guided by Umesh Yadav Sir (EDC IIT Delhi)")
st.markdown("📊 Enterprise AI System | HR Analytics | Machine Learning Dashboard")