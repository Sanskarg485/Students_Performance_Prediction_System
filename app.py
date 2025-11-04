# ============================================
# app.py - Student Performance Dashboard
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import StudentPerformanceModel

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Teacher Dashboard - Student Performance",
    page_icon="🎓",
    layout="wide",
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    return StudentPerformanceModel()

model_obj = load_model()

# ============================================
# HEADER
# ============================================
st.title("🎓 Teacher Dashboard - Student Performance Analysis")
st.markdown(
    "Upload a CSV file to analyze student data, predict performance levels, "
    "and view insights interactively."
)

# ============================================
# FILE UPLOAD SECTION
# ============================================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # --- Data Overview ---
    st.subheader("📊 Data Overview")
    st.dataframe(data.head(10))

    # --- Basic Info ---
    st.markdown("### 🧩 Basic Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(data))
    col2.metric("Features", data.shape[1])
    col3.metric("Missing Values", data.isnull().sum().sum())

    # --- Prediction ---
    try:
        data = model_obj.predict(data)
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        st.stop()

    st.markdown("### 🎯 Prediction Results")
    st.dataframe(
        data["Predicted_Performance"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Performance Level", "Predicted_Performance": "Count"})
    )

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Performance Insights",
            "🔍 Feature Correlation",
            "🔥 Feature Importance",
            "📊 Student-Level Analysis",
        ]
    )

    # --- TAB 1: Performance Insights ---
    with tab1:
        st.subheader("📈 Overall Performance Distribution")
        fig = px.histogram(
            data,
            x="Predicted_Performance",
            color="Predicted_Performance",
            title="Predicted Student Performance Levels",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Average Feature Values by Performance")
        avg_df = data.groupby("Predicted_Performance").mean(numeric_only=True)
        st.dataframe(avg_df)
        fig2 = px.bar(
            avg_df.T,
            barmode="group",
            title="Average Feature Values by Performance",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- TAB 2: Correlation Heatmap ---
    with tab2:
        st.subheader("🔍 Feature Correlation Heatmap")
        corr = data.select_dtypes(include="number").corr()
        fig3 = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        st.plotly_chart(fig3, use_container_width=True)

    # --- TAB 3: Feature Importance ---
    with tab3:
        st.subheader("🔥 Feature Importance (Model-Based)")
        imp_df = model_obj.get_feature_importance(data)
        if not imp_df.empty:
            fig4 = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Feature importance not available for this model type.")

    # --- TAB 4: Student-Level Analysis ---
    with tab4:
        st.subheader("📊 Individual Student Insights")

        # Detect student identifier column
        id_col = None
        for col in data.columns:
            if "id" in col.lower():
                id_col = col
                break

        if id_col:
            selected_student = st.selectbox(
                "Select a Student ID:",
                data[id_col].unique(),
                index=0,
            )

            # Filter data for selected student
            student_data = data[data[id_col] == selected_student]

            st.markdown("### 🎓 Selected Student Details")
            st.dataframe(student_data)

            # Prepare data for visualization
            numeric_features = [
                col
                for col in student_data.select_dtypes(include="number").columns
                if col not in ["Predicted_Performance"]
            ]
            plot_data = student_data[numeric_features].melt(
                var_name="Feature", value_name="Values"
            )

            st.markdown("### 📊 Selected Student Feature Values")
            fig5 = px.bar(
                plot_data,
                x="Feature",
                y="Values",
                color="Feature",
                text="Values",
                color_discrete_sequence=px.colors.qualitative.Set3,
                title=f"Performance Breakdown for {selected_student}",
            )
            fig5.update_traces(textposition="outside")
            fig5.update_layout(
                showlegend=True,
                xaxis_title=None,
                yaxis_title="Values",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig5, use_container_width=True)

            # Show performance prediction
            perf = student_data["Predicted_Performance"].iloc[0]
            st.success(f"🧠 Predicted Performance Level for **{selected_student}**: {perf}")

        else:
            st.warning("⚠️ Student ID column not found in uploaded dataset.")

else:
    st.info("👆 Please upload a CSV file to begin analysis.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("Developed with ❤️ for Teachers | AI-Powered Student Insights Dashboard")
