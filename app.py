# ==========================================================
# 🚀 CUSTOMER SEGMENTATION INTELLIGENCE APP
# ==========================================================
# PURPOSE:
# End-to-end business + ML storytelling app
# Combines:
# - RFM segmentation (business rules)
# - KMeans clustering (ML)
# - Explainability (SHAP-like)
# - Business insights & strategy
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# ⚙️ PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="RFM Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("RFM Intelligence Dashboard")
st.warning("🌙 For best experience, view this dashboard in **dark mode** — visuals are optimized for dark backgrounds.")

# ==========================================================
# 📥 LOAD DATA & MODELS
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("rfm_clustered_output.csv")
    explain_df = pd.read_csv("cluster_explainability.csv")
    return df, explain_df

@st.cache_resource
def load_models():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    cluster_labels = pickle.load(open("cluster_labels.pkl", "rb"))
    return scaler, kmeans, cluster_labels

df, explain_df = load_data()
scaler, kmeans, cluster_labels = load_models()

# ==========================================================
# 🧩 TWO MAIN TABS ONLY
# ==========================================================
tab1, tab2 = st.tabs([
    "📊 Analysis & Insights",
    "🔮 Prediction & Strategy"
])

# ==========================================================
# 📊 TAB 1: ANALYSIS & INSIGHTS
# ==========================================================
with tab1:

    
    st.markdown(
    "<h2 style='color:#FFD700;'>📊 Segmentation Analysis (RFM vs KMeans)</h2>",
    unsafe_allow_html=True
)
    st.info("""
    This section compares **rule-based RFM segmentation** with **machine learning clustering (K-Means)**.
    
    🔍 What you’ll learn here:
    - How customers are grouped using business rules vs data-driven methods  
    - Why some customers are classified differently (mismatch analysis)  
    - Which clusters drive the most revenue  
    - Behavioral patterns of customers (recency, frequency, spending)  
    - Why the model grouped customers the way it did (explainability)  
    
    💡 Goal: Understand customer behavior deeply and identify actionable business insights.
    """)
    st.divider()
    # -------------------------------
    # 🍩 Donut Chart (Cluster Share)
    # -------------------------------

    # ------------------------------------------
    # 🔀 COMPARISON
    # ------------------------------------------
    if "Customer_Segment" in df.columns:
        comparison = pd.crosstab(df["Customer_Segment"], df["Cluster"])

        
        st.markdown(
    "<h3 style='color:#ff7f0e;'>1. Segment Mapping </h3>",
    unsafe_allow_html=True
)
        st.dataframe(comparison)
    


    comparison = pd.crosstab(df["Customer_Segment"], df["Cluster"])
    
    fig = px.imshow(
        comparison,
        text_auto=True,
        aspect="auto",
        title="RFM vs KMeans Mapping"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # ------------------------------------------
    # 🧠 WHY 5 SEGMENTS → 2 CLUSTERS
    # ------------------------------------------

    st.info("""
🧠 Why 5 Segments Became 2 Clusters?
    
    • RFM segmentation is rule-based → creates multiple predefined groups  
    • KMeans is data-driven → groups customers based on similarity  

    👉 Result:
    Multiple RFM segments collapse into fewer natural clusters  

    💡 Meaning:
    - Some segments behave similarly
    - Customers are not as different as rules suggest
    """)

   # ==========================================================
    # 💰 REVENUE CONTRIBUTION (ENHANCED)
    # ==========================================================
    
    st.markdown(
    "<h3 style='color:#ff7f0e;'>2. Revenue Contribution </h3>",
    unsafe_allow_html=True
)

    revenue = df.groupby("Cluster")["Monetary"].sum()
    
    for c in revenue.index:
        st.write(f"Cluster {c} → {'🟩'*int(revenue[c]/revenue.max()*20)}")
# revenue bar Hover enabled
    

    revenue = df.groupby("Cluster")["Monetary"].sum().reset_index()
    
    fig = px.bar(
        revenue,
        x="Cluster",
        y="Monetary",
        title="Revenue by Cluster",
        text_auto=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

    
# 📈 CUSTOMER BEHAVIOR SCATTER (ADD HERE)

    
    
    fig = px.scatter(
        df,
        x="Recency",
        y="Monetary",
        color="Cluster",
        size="Frequency",
        hover_data=["Customer_Segment","Segment_KMeans"],
        title="Customer Distribution by Behavior (Recency vs Monetary)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 🧭 CLUSTER PROFILE (RADAR CHART)

    #st.subheader("🧭 Cluster Profiles (Behavior Summary)")
    
    #cluster_summary = df.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean().reset_index()
    
    #for _, row in cluster_summary.iterrows():
    
   #     fig = go.Figure()
    
   #     fig.add_trace(go.Scatterpolar(
    #        r=[row["Recency"], row["Frequency"], row["Monetary"]],
    #        theta=["Recency","Frequency","Monetary"],
     #       fill='toself',
    #        name=f"Cluster {int(row['Cluster'])}"
     #   ))
    
     #   fig.update_layout(
      #      polar=dict(radialaxis=dict(visible=True)),
       #     title=f"Cluster {int(row['Cluster'])} Profile"
      #  )
    
      #  st.plotly_chart(fig)
    # ------------------------------------------
    # ⚠️ MISMATCH ANALYSIS
    # ------------------------------------------
    
    st.markdown(
    "<h3 style='color:#ff7f0e;'>3. Misclassified Customers </h3>",
    unsafe_allow_html=True
)

    mismatch = df[df["Customer_Segment"] != df["Segment_KMeans"]]

    st.write(f"Total mismatched customers: {len(mismatch)}")

    sample = mismatch.head(10)
    st.dataframe(sample)

    # ------------------------------------------
    # 🧠 WHY MISMATCH HAPPENS (EXPLAIN)
    # ------------------------------------------

    st.info("""
    🧠 Why Mismatch Happens?
    
    • RFM uses strict score rules  
    • KMeans uses distance-based similarity  

    👉 Example:
    - A customer marked "Loyal" may behave like "At Risk"  
    - KMeans captures actual behavior patterns  

    💡 Insight:
    ML corrects rigid business rules
    """)

   
 # ------------------------------------------
 # 👤 CUSTOMER LEVEL EXPLANATION
 # ------------------------------------------
    
    st.markdown(
    "<h3 style='color:#ff7f0e;'>4. Customer-Level Explanation</h3>",
    unsafe_allow_html=True
)

    cust_id = st.selectbox("Select Customer Index", df.index)

    row = df.loc[cust_id]
    exp = explain_df.loc[cust_id]

    
    st.write(row[["Recency","Frequency","Monetary"]])

    st.write(f"RFM Segment: {row['Customer_Segment']}")
    st.write(f"KMeans Cluster: {row['Cluster']} ({row['Segment_KMeans']})")

    
    fig = px.bar(
        x=exp[["Recency","Frequency","Monetary"]],
        y=["Recency","Frequency","Monetary"],
        orientation='h',
        title="Feature Contribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 🔮 TAB 2: PREDICTION & STRATEGY
# ==========================================================
with tab2:

    
    st.markdown(
    "<h2 style='color:#FFD700;'>🔮 Predict Customer Segment & Strategy</h2>",
    unsafe_allow_html=True
)
    st.info("""
    This section allows you to **predict customer segments in real-time** using the trained clustering model.
    
    🔮 What you can do here:
    - Input customer behavior (Recency, Frequency, Monetary)  
    - Predict which cluster the customer belongs to  
    - Understand *why* the model assigned this segment  
    - Get **business recommendations** to engage the customer effectively  
    
    💡 Goal: Turn data into actionable decisions for marketing, retention, and growth strategies.
    """)
    st.divider()
    # ------------------------------------------
    # INPUT
    # ------------------------------------------
    
    st.markdown(
    "<h3 style='color:#ff7f0e;'>Input Customer Details </h3>",
    unsafe_allow_html=True
)
    recency = st.number_input("Recency", value=10)
    frequency = st.number_input("Frequency", value=5)
    monetary = st.number_input("Monetary", value=1000)

    if st.button("Predict"):

        new_data = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(new_data)[0]
        segment = cluster_labels[cluster]
        
        
        st.markdown(
    "<h3 style='color:#ff7f0e;'>Predicted Cluster </h3>",
    unsafe_allow_html=True
)
        st.success(f"Cluster: {cluster}")
        st.success(f"Segment: {segment}")
        
           
       # PREDICTION RESULT → GAUGE STYLE (SIMULATED)
      
    
        score = (frequency + monetary - recency)
        st.progress(min(max(int(score), 0), 100))
        
        # --------------------------------------
        # 🧠 EXPLANATION
        # --------------------------------------
        
        st.markdown(
    "<h3 style='color:#ff7f0e;'>🧠 Why this Cluster? </h3>",
    unsafe_allow_html=True
)

        if recency < df["Recency"].mean():
            st.write("✅ Customer is recent")
        else:
            st.write("⚠️ Customer is inactive")

        if frequency > df["Frequency"].mean():
            st.write("✅ High engagement")
        else:
            st.write("⚠️ Low engagement")

        if monetary > df["Monetary"].mean():
            st.write("💰 High value")
        else:
            st.write("💸 Low value")
            
         # --------------------------------------
        # Shows where customer lies
        # --------------------------------------    
        st.markdown(
    "<h3 style='color:#ff7f0e;'>Customer Position </h3>",
    unsafe_allow_html=True
)
    
        fig = px.scatter(
            df,
            x="Recency",
            y="Monetary",
            color="Cluster",
            opacity=0.3
        )
        
        fig.add_scatter(
            x=[recency],
            y=[monetary],
            mode='markers',
            marker=dict(size=15, color='yellow'),
            name="New Customer"
        )
        
        st.plotly_chart(fig, use_container_width=True)
            
    # --------------------------------------
        # Shows deviation from average
        # --------------------------------------  
        st.markdown(
    "<h3 style='color:#ff7f0e;'>📊 Customer vs Average </h3>",
    unsafe_allow_html=True
)
        
        avg = df[["Recency","Frequency","Monetary"]].mean()
        
        comp_df = pd.DataFrame({
            "Feature": ["Recency","Frequency","Monetary"],
            "Customer": [recency, frequency, monetary],
            "Average": avg.values
        })
        
        fig = px.bar(
            comp_df,
            x="Feature",
            y=["Customer","Average"],
            barmode="group",
            title="Feature Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --------------------------------------
        # 💼 BUSINESS STRATEGY
        # --------------------------------------
        st.markdown(
    "<h3 style='color:#ff7f0e;'>💼 Recommended Strategy for Business </h3>",
    unsafe_allow_html=True
)

        if "Champion" in segment or "Loyal" in segment:
            st.success("""
            • Reward loyalty  
            • VIP programs  
            • Early access offers  
            """)

        elif "Risk" in segment:
            st.warning("""
            • Retention campaigns  
            • Discount offers  
            • Re-engagement emails  
            """)

        else:
            st.info("""
            • Upsell / cross-sell  
            • Product recommendations  
            • Engagement campaigns  
            """)