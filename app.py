import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation using K-Means",
    page_icon="👥",
    layout="wide"
)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/Mall_Customers.csv")

df = load_data()

# ---------------------------------------------------------
# Train K-Means Model
# ---------------------------------------------------------
features = df[["Annual Income (k$)", "Spending Score (1-100)"]]

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(features)

# Cluster-to-segment mapping used for business interpretation
segment_names = {
    0: "High-Income High-Spending Customers",
    1: "High-Income Low-Spending Customers",
    2: "Low-Income High-Spending Customers",
    3: "Average-Income Average-Spending Customers",
    4: "Low-Value Customers"
}

df["Segment"] = df["Cluster"].map(segment_names)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("👥 Customer Segmentation Using K-Means Clustering")
st.markdown(
    "An interactive customer segmentation application using "
    "Annual Income and Spending Score."
)

st.divider()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("Customer Information")

income = st.sidebar.slider(
    "Annual Income (k$)",
    min_value=int(df["Annual Income (k$)"].min()),
    max_value=int(df["Annual Income (k$)"].max()),
    value=60
)

spending = st.sidebar.slider(
    "Spending Score (1-100)",
    min_value=int(df["Spending Score (1-100)"].min()),
    max_value=int(df["Spending Score (1-100)"].max()),
    value=50
)

predict_button = st.sidebar.button("Predict Customer Segment")

# ---------------------------------------------------------
# Dashboard Metrics
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(df))
col2.metric("Number of Clusters", 5)
col3.metric("Average Income", f"${df['Annual Income (k$)'].mean():.1f}k")
col4.metric("Average Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")

st.divider()

# ---------------------------------------------------------
# Customer Segment Prediction
# ---------------------------------------------------------
st.subheader("🎯 Customer Segment Prediction")

if predict_button:
    prediction = kmeans.predict([[income, spending]])[0]
    segment = segment_names[prediction]

    st.success(f"Predicted Segment: **{segment}**")
    st.write(f"Annual Income: **${income}k**")
    st.write(f"Spending Score: **{spending}**")

else:
    st.info("Select income and spending score from the sidebar, then click Predict Customer Segment.")

# ---------------------------------------------------------
# Cluster Visualization
# ---------------------------------------------------------
st.subheader("📊 Customer Segmentation")

fig, ax = plt.subplots(figsize=(10, 6))

for cluster in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster]

    ax.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=segment_names[cluster],
        alpha=0.7
    )

ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    label="Cluster Centroids"
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segmentation using K-Means Clustering")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

st.pyplot(fig)

# ---------------------------------------------------------
# Cluster Summary
# ---------------------------------------------------------
st.subheader("📋 Customer Segment Summary")

summary = (
    df.groupby("Segment")
    .agg(
        Customers=("CustomerID", "count"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending_Score=("Spending Score (1-100)", "mean")
    )
    .round(2)
    .sort_values("Customers", ascending=False)
)

st.dataframe(summary, use_container_width=True)

# ---------------------------------------------------------
# Business Recommendations
# ---------------------------------------------------------
st.subheader("💡 Business Recommendations")

recommendations = {
    "High-Income High-Spending Customers":
        "Use premium offers, loyalty programs, personalized promotions, and upselling strategies.",
    "High-Income Low-Spending Customers":
        "Use targeted engagement campaigns, personalized recommendations, and incentives to increase spending.",
    "Low-Income High-Spending Customers":
        "Offer affordable premium products, promotions, and loyalty rewards to maintain engagement.",
    "Average-Income Average-Spending Customers":
        "Use regular promotions, cross-selling, and personalized campaigns to increase customer value.",
    "Low-Value Customers":
        "Use introductory offers and engagement campaigns to encourage higher spending."
}

for segment, recommendation in recommendations.items():
    st.markdown(f"**{segment}:** {recommendation}")

# ---------------------------------------------------------
# Dataset Preview
# ---------------------------------------------------------
with st.expander("🔍 View Customer Dataset"):
    st.dataframe(df, use_container_width=True)

st.divider()

st.caption(
    "Customer Segmentation Project | K-Means Clustering | "
    "Data Science | Machine Learning | Business Analytics"
)
