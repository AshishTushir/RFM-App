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
import base64

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}" 
    width="100%" height="700px" type="application/pdf">
    </iframe>
    """

    import streamlit as st
    st.markdown(pdf_display, unsafe_allow_html=True)
# ==========================================================
# ⚙️ PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="RFM Intelligence Dashboard 🤖",
    page_icon="📊",
    layout="wide"
)

st.title("RFM Intelligence Dashboard 🤖")
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

# LOAD DATA FROM GITHUB
    # -------------------------------
    
@st.cache_data
def load_data():
    #url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/RFM_data.csv"
    #return pd.read_csv(url)
    rfm_df = pd.read_csv("RFM_data.csv", encoding="latin1")
    return rfm_df
    # 🔥 ADD THIS LINE
rfm_df = load_data()


# ==========================================================
# 🧩 Four MAIN TABS ONLY
# ==========================================================
tab0, tab1, tab2, tab3 = st.tabs([
    "📄 Project Summary & Proof of Work",
    "🎯 Power BI Insights & Strategy",
    "🤖 ML Analysis & Insights",
    "🔮 Prediction & Strategy"
    
    
])
# ==========================================================
# 🔮 TAB 0: Project Summary
# ==========================================================
with tab0:
    st.markdown(
    "<h2 style='color:#FFD700;'>📄 Project Overview</h2>",
    unsafe_allow_html=True
)
    st.info("""
This section provides a complete overview of the project.

👉 Includes:
- End-to-end pipeline (SQL → BI → ML → App)
- Business insights and segmentation logic
- Machine learning validation
- Final conclusions and strategy

📌 This document serves as the **single source of truth** for the project.
""")
    
   
    # Convert image to base64
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
        
    img_base64 = get_base64_image("images/FinalPipeline.png")
        
        # Display with styled container
    st.markdown(f"""
        <div style="
            border: 4px solid #FFD700;
            border-radius: 14px;
            padding: 18px;
            margin-top: 25px;
            margin-bottom: 35px;
            background-color: #111111;
            box-shadow: 0 0 25px rgba(255, 215, 0, 0.6);
        ">
            <img src="data:image/png;base64,{img_base64}" 
                 style="width:100%; border-radius:20px;">
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(
    "<h3 style='color:#ff7f0e;'>1. Project Report</h3>",
    unsafe_allow_html=True
)

    # ===============================
    # 📄 MAIN PDF VIEWER
    # ===============================
    display_pdf("pdfs/project_report.pdf")

    # ===============================
    # ⬇️ DOWNLOAD SUMMARY (NEW)
    # ===============================
   

    st.markdown("---")

    # ===============================
    # OTHER DOWNLOADS
    # ===============================
    
    st.markdown(
    "<h3 style='color:#ff7f0e;'>2. Proof of Work ⬇️ </h3>",
    unsafe_allow_html=True
)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "🗄️ SQL Script",
            data=open("SQL/report_customers.sql", "rb"),
            file_name="report_customers.sql"
        )

        st.download_button(
            "📊 Power BI Dashboard (.pbix)",
            data=open("Power BI Report/RFM Analysis.pbix", "rb"),
            file_name="RFM Analysis.pbix"
        )
        st.markdown("""
    🔗 **Connect**  
    👉 tushirgetsmail@gmail.com
    """)
    with col2:
        st.download_button(
            "🤖 ML Modeling",
            data=open("Modelling/RFM Model.ipynb", "rb"),
            file_name="RFM Model.ipynb"
        )
        st.download_button(
        label="📥 Download Full Project Summary (PDF)",
        data=open("pdfs/project_report.pdf", "rb"),
        file_name="project_report.pdf"
    )
        st.markdown("""
🔗 **GitHub Repository**  
👉 https://github.com/AshishTushir/RFM-App
""")

    st.markdown("---")

    st.caption("📌 End-to-end transparency: data → insights → modeling → deployment")
# ================================
# TAB 1: BUSINESS INSIGHTS (RFM)
# ================================
with tab1:

    # -------------------------------
    # TITLE + INTRO
    # -------------------------------

    st.markdown(
    "<h2 style='color:#FFD700;'>🎯 Business Insights & Market Targeting</h2>",
    unsafe_allow_html=True
)

    st.info("""
    This section explains customer segmentation using rule-based RFM scoring derived from business logic.
    👉 Focus:
    - Understand customer behavior using Recency, Frequency, Monetary
    - Identify revenue-driving segments
    - Design targeted marketing strategies
    """)

# -------------------------------
# RULE-BASED SEGMENTATION TABLE
# -------------------------------

    rfm_rules = pd.DataFrame({
        "Segment": [
            "🏆 Champions",
            "⭐ Loyal Customers",
            "💰 Big Spenders",
            "⚠️ At Risk",
            "❌ Lost",
            "🔹 Others"
        ],
        
        "Recency (R)": [
            "≤ 2",
            "≤ 3",
            "≥ 3",
            "≥ 3",
            "= 5",
            "-"
        ],
        
        "Frequency (F)": [
            "≤ 2",
            "≤ 2",
            "-",
            "≤ 3",
            "= 5",
            "-"
        ],
        
        "Monetary (M)": [
            "≤ 2",
            "-",
            "≤ 2",
            "-",
            "= 5",
            "-"
        ],
        
        "Description": [
            "Best customers across all metrics",
            "Frequent and recent buyers",
            "High spenders but not recent",
            "Previously active, now inactive",
            "Low engagement across all metrics",
            "Remaining customers"
        ]
    })

    st.table(rfm_rules)
    # -------------------------------
    # KPI CARDS
    # -------------------------------
    # FILTER (SEGMENT SELECTOR)
    segments = sorted(rfm_df['Customer_Segment'].dropna().unique())
    selected_segments = st.multiselect(
        "Select Segment(s)",
        options=segments,
        default=segments
    )
    # Apply filter
    filtered_df = rfm_df[rfm_df['Customer_Segment'].isin(selected_segments)]

    # -------------------------------
    # POWER BI DASHBOARD IMAGE
    # -------------------------------
    

    
        # Convert image to base64
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
        
    img_base64 = get_base64_image("images/rfm_power_1.png")
        
        # Display with styled container
    st.markdown(f"""
        <div style="
            border: 4px solid #FFD700;
            border-radius: 14px;
            padding: 18px;
            margin-top: 25px;
            margin-bottom: 35px;
            background-color: #111111;
            box-shadow: 0 0 25px rgba(255, 215, 0, 0.6);
        ">
            <img src="data:image/png;base64,{img_base64}" 
                 style="width:100%; border-radius:10px;">
        </div>
        """, unsafe_allow_html=True)
    # -------------------------------
    # REVENUE BY SEGMENT (BAR)
    # -------------------------------
   
    st.markdown(
    "<h3 style='color:#ff7f0e;'> 1. Revenue Contribution by Segment </h3>",
    unsafe_allow_html=True
)

    
    #
    
    segment_revenue = (
        filtered_df.groupby('Customer_Segment', as_index=False)['Monetary']
        .sum()
    )

    fig_bar = px.bar(
        segment_revenue,
        x='Customer_Segment',
        y='Monetary',
        text_auto=True,
        title="Revenue Distribution Across Customer Segments"
    )

    fig_bar.update_layout(
        template="plotly_dark",
        xaxis_title="Customer Segment",
        yaxis_title="Total Revenue"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    
    # PIE CHART (REVENUE SHARE)
    
    fig_pie = px.pie(
        segment_revenue,
        names='Customer_Segment',
        values='Monetary',
        title="Revenue Share by Segment"
    )

    fig_pie.update_layout(template="plotly_dark")

    st.plotly_chart(fig_pie, use_container_width=True)


    # -------------------------------
    # KEY INSIGHTS
    # -------------------------------
   
    st.markdown(
    "<h3 style='color:#ff7f0e;'> 2. Key Business Insightss </h3>",
    unsafe_allow_html=True
)

    st.success("""
• Loyal Customers form the largest segment → strong retention base  
• Big Spenders generate high revenue despite lower frequency  
• At Risk customers show high recency → potential churn  
• Champions are fewer but extremely valuable  

👉 Insight:
Revenue is concentrated among high-value customers.
""")

    # -------------------------------
    # STRATEGY TABLE (DYNAMIC)
    # -------------------------------
   
    st.markdown(
    "<h3 style='color:#ff7f0e;'>  3. Strategy Summary Tables </h3>",
    unsafe_allow_html=True
)
    import pandas as pd
    
    strategy_data = {
        "Category": ["Target", "Why?", "Strategy", "Goal"],
    
        "🎯 Loyalty Programs": [
            "Loyal Customers",
            "High frequency → engaged users\nClose to becoming Champions",
            "Rewards programs\nMembership tiers\nEarly access",
            "Increase spending → convert to Champions"
        ],
    
        "💸 Smart Discount Strategy": [
            "At Risk + Big Spenders",
            "High recency → not purchasing recently\nHigh value but low engagement",
            "Personalized discounts\nLimited-time offers",
            "Reactivate customers before churn"
        ],
    
        "🔁 Retargeting Campaigns": [
            "At Risk + Lost",
            "Low engagement\nCustomers drifting away",
            "Email campaigns\nAds + notifications",
            "Bring customers back"
        ]
    }
    
    strategy_df = pd.DataFrame(strategy_data)
    
    st.dataframe(strategy_df, use_container_width=True)

    # -------------------------------
    # BUSINESS VALUE
    # -------------------------------
   
    st.markdown(
    "<h3 style='color:#ff7f0e;'> 4. Why RFM Works 🏆</h3>",
    unsafe_allow_html=True
)

    st.info("""
• Simple and interpretable  
• Business-friendly segmentation  
• Enables quick targeting  
• Identifies high-value customers  
""")

    # -------------------------------oard in form of information to a separate header
    # LIMITATIONst below the imageof power bi dashb
    # -------------------------------tion ju
    
    st.markdown(
    "<h3 style='color:#ff7f0e;'> 5. Limitation of Rule-Based Approach ⚠️ </h3>",
    unsafe_allow_html=True
)

    st.warning("""
• Fixed rules → rigid segmentation  
• Cannot capture hidden behavior  
• May misclassify customers  
""")

    # -------------------------------
    # TRANSITION TO ML
    # -------------------------------
    
   


    st.success("""🤖 Moving to Machine Learning 
    
To validate segmentation and uncover hidden patterns, we use K-Means clustering in the next tab.

👉 This answers: Are we segmenting customers correctly?
""")
    
# ==========================================================
# 📊 TAB 2: ANALYSIS & INSIGHTS
# ==========================================================
with tab2:

    
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
# 🔮 TAB 3: PREDICTION & STRATEGY
# ==========================================================
with tab3:

    
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
