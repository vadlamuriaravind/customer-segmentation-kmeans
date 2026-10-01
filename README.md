# Customer Segmentation Using K-Means Clustering

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-segmentation-kmeans-s2ewbtaskf9aryzluogpmd.streamlit.app/)

An end-to-end unsupervised machine learning project that segments mall customers using K-Means clustering based on Annual Income and Spending Score.

## Table of Contents

- [Live Demo](#live-demo)
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Project Workflow](#project-workflow)
- [Machine Learning Approach](#machine-learning-approach)
- [Results](#results)
- [Cluster Profiles](#cluster-profiles)
- [Business Recommendations](#business-recommendations)
- [Model Evaluation](#model-evaluation)
- [Streamlit Application](#streamlit-application)
- [Installation and Usage](#installation-and-usage)
- [Visualizations](#visualizations)
- [Key Learning Outcomes](#key-learning-outcomes)
- [Author](#author)

## Live Demo

👉 **[Customer Segmentation Streamlit App](https://customer-segmentation-kmeans-s2ewbtaskf9aryzluogpmd.streamlit.app/)**

## Project Overview

This project applies unsupervised machine learning to identify meaningful groups of mall customers based primarily on their **Annual Income** and **Spending Score**.

The objective is to transform raw customer data into meaningful customer segments that support data-driven business strategies, including targeted marketing campaigns, personalized promotions, customer engagement, customer retention, cross-selling, upselling, and premium customer strategies.

The project demonstrates a complete Data Science workflow — from raw data analysis and exploratory analysis through machine learning, business interpretation, and interactive deployment using Streamlit.

## Business Problem

Businesses serve customers with different income levels, spending behaviors, and purchasing characteristics. A single marketing strategy is therefore not equally effective for every customer.

Customer segmentation helps businesses identify groups of customers with similar characteristics and develop more relevant strategies for each group. This project uses K-Means Clustering to identify customer groups based on income and spending behavior and to transform the resulting clusters into meaningful business insights.

## Project Objectives

The main objective is to analyze mall customer data and identify meaningful customer segments using K-Means Clustering.

The project covers:

- Data quality analysis
- Exploratory Data Analysis
- Feature selection and feature scaling
- Cluster evaluation using the Elbow Method and Silhouette Score
- Customer segmentation and cluster profiling
- Business interpretation and recommendations
- Results export
- Streamlit deployment

## Dataset

- **Dataset:** Mall Customer Segmentation Data
- **Source:** [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Records:** 200 customers
- **Attributes:** 5 original columns

### Dataset Features

| Feature | Role | Description |
| --- | --- | --- |
| CustomerID | Identifier | Unique identifier for each customer |
| Gender | Demographic | Customer gender |
| Age | Demographic | Customer age |
| Annual Income (k$) | Financial | Customer annual income |
| Spending Score (1-100) | Behavioral | Customer spending behavior score |

The primary features used for the final clustering analysis are **Annual Income (k$)** and **Spending Score (1-100)**. These features help identify customers with similar financial capacity and spending behavior.

**Note:** Spending Score is a composite index supplied by the dataset. It should not be interpreted as a purchase amount, profit contribution, or customer lifetime value.

## Technologies Used

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (KMeans, StandardScaler, silhouette_score)
- **Environment:** Google Colab / Jupyter Notebook
- **Deployment:** Streamlit
- **Version control:** Git and GitHub

## Project Structure

```text
customer-segmentation-kmeans/
│
├── data/
│   └── Mall_Customers.csv
├── images/
│   └── Project visualizations and charts
├── Results/
│   └── Customer_Segmentation_Results.csv
├── Customer_Segmentation_Project.ipynb
├── app.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

- `data/` — the original customer dataset
- `images/` — exploratory analysis, clustering, and model evaluation visualizations
- `Results/` — the final customer segmentation results
- `Customer_Segmentation_Project.ipynb` — the complete end-to-end implementation
- `app.py` — the interactive Streamlit application
- `requirements.txt` — required Python libraries
- `LICENSE` — MIT license
- `README.md` — project documentation

## Project Workflow

```text
1.  Data Loading
2.  Data Understanding
3.  Data Quality Analysis
4.  Exploratory Data Analysis
5.  Feature Selection
6.  Feature Scaling
7.  Elbow Method
8.  Silhouette Score
9.  K-Means Clustering
10. Cluster Profiling
11. Customer Segment Analysis
12. Business Recommendations
13. Model Evaluation
14. Results Export
15. Streamlit Deployment
```

## Machine Learning Approach

### Algorithm

**K-Means Clustering**, an unsupervised algorithm that groups similar data points into clusters. Customers are grouped by similarity in Annual Income and Spending Score.

### Feature Scaling

K-Means is distance-based, so features are standardized with **StandardScaler** before clustering. This prevents a feature with a larger numerical range from disproportionately influencing the distance calculations.

### Cluster Selection

The number of clusters is evaluated using two complementary diagnostics:

- **Elbow Method** — analyzes Within-Cluster Sum of Squares (WCSS) across candidate values of K
- **Silhouette Score** — evaluates cluster cohesion and separation

Business interpretability is treated as a third criterion: a solution must be describable in business terms to be useful.

### Final Configuration

| Parameter | Value |
| --- | --- |
| Algorithm | K-Means Clustering |
| n_clusters | 5 |
| random_state | 42 |
| n_init | 10 |
| Features | Annual Income (k$), Spending Score (1-100) |
| Preprocessing | StandardScaler |
| WCSS at K=5 | 65.5684 |
| Silhouette Score at K=5 | 0.5547 |

## Results

The final model assigns all 200 customers across five technical clusters:

| Technical Cluster | Customer Count |
| --- | --- |
| Cluster 0 | 81 |
| Cluster 1 | 39 |
| Cluster 2 | 22 |
| Cluster 3 | 35 |
| Cluster 4 | 23 |
| **Total** | **200** |

Cluster identifiers are technical model outputs and carry no inherent business meaning. Business significance is assigned during profiling and interpretation.

Results are exported to:

```text
Customer_Segmentation_Results.csv
```

The file contains the original customer information plus two model-derived fields — **Cluster** (technical assignment) and **Segment** (business interpretation).

## Cluster Profiles

| Cluster | Customers | Avg Age | Avg Income (k$) | Avg Spending Score |
| --- | --- | --- | --- | --- |
| Cluster 0 | 81 | 42.72 | 55.30 | 49.52 |
| Cluster 1 | 39 | 32.69 | 86.54 | 82.13 |
| Cluster 2 | 22 | 25.27 | 25.73 | 79.36 |
| Cluster 3 | 35 | 41.11 | 88.20 | 17.11 |
| Cluster 4 | 23 | 45.22 | 26.30 | 20.91 |

Five technical clusters map to **four** business segments. Cluster 0 and Cluster 1 both satisfy the High-Value condition under the implemented median-based mapping rule, so they share that label — even though their numerical profiles differ.

| Business Segment | Technical Clusters | Customers |
| --- | --- | --- |
| High-Value Customers | Cluster 0, Cluster 1 | 120 |
| High-Income Low-Spending Customers | Cluster 3 | 35 |
| Low-Income High-Spending Customers | Cluster 2 | 22 |
| Low-Value Customers | Cluster 4 | 23 |

These labels are analytical descriptors for clusters, not measured profitability or customer lifetime value.

## Business Recommendations

| Segment | Recommended Strategy |
| --- | --- |
| High-Value Customers | Premium offers, loyalty rewards, personalized recommendations, and exclusive products to strengthen retention |
| High-Income Low-Spending Customers | Personalized promotions, relevant product recommendations, and targeted incentives to increase engagement and spending |
| Low-Income High-Spending Customers | Loyalty programmes, affordable bundles, repeat-purchase incentives, and appropriately priced offers to sustain engagement |
| Low-Value Customers | Cost-effective promotions, introductory offers, and personalized campaigns to improve engagement incrementally |

These recommendations are derived from observed patterns in the supplied dataset. They are analytical decision support, not guarantees of campaign performance.

## Model Evaluation

Because this is an **unsupervised** learning task, conventional classification metrics such as accuracy, precision, recall, and F1-score do not apply — there is no ground-truth segment label against which predictions could be compared.

The model is evaluated using:

| Dimension | Measure | Result |
| --- | --- | --- |
| Technical quality | WCSS at K=5 | 65.5684 |
| Technical quality | Silhouette Score at K=5 | 0.5547 |
| Structural | Cluster size distribution | 81 / 39 / 22 / 35 / 23 |
| Structural | Empty or degenerate clusters | None |
| Structural | Visual separation | Confirmed by scatter plot |
| Business | Segment describability | Four interpretable segments |
| Business | Actionability | Recommendations developed per segment |

The Silhouette Score is an internal validation measure. It ranges from −1 to +1, where higher values indicate better cluster cohesion and separation. It is **not** a classification accuracy percentage.

## Streamlit Application

The interactive application allows users to:

- Enter a customer's Annual Income and Spending Score using sidebar sliders
- Predict the customer's cluster and view the corresponding business segment
- View summary metrics for the dataset
- Explore the segmentation scatter plot with cluster centroids
- Review the technical cluster profile and business segment summary
- Read segment-specific business recommendations
- Download the segmentation results as a CSV file

The application applies the **same fitted StandardScaler** used during training, so predictions are made in the same feature space as the notebook. Cluster centroids are inverse-transformed back to the original income and spending scale before plotting.

**What the prediction means:** the output is a cluster assignment based on similarity in the modelled feature space. It is not a forecast of future spending, a measure of customer value, or an output of a trained recommendation engine.

## Installation and Usage

**1. Clone the repository**

```bash
git clone [github.com](https://github.com/vadlamuriaravind/customer-segmentation-kmeans.git)
```

**2. Navigate to the project directory**

```bash
cd customer-segmentation-kmeans
```

**3. Install the required libraries**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```



## Key Learning Outcomes

This project demonstrates practical experience in:

- Python programming and data analysis
- Data quality assessment and Exploratory Data Analysis
- Data visualization with Matplotlib and Seaborn
- Feature selection and feature scaling for distance-based clustering
- Unsupervised machine learning with K-Means Clustering
- Cluster optimization using the Elbow Method and Silhouette Score
- Cluster profiling and business interpretation
- Business analytics and recommendation development
- Streamlit application deployment
- GitHub project organization and documentation



## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.





Visualizations

Exploratory Data Analysis

[Age vs Annual Income](images/Age%20vs%20Annual%20Income.png)

[Age vs Spending Score](images/Age%20vs%20Spending%20Score.png)

[Annual Income Distribution of Customers](images/Annual%20Income%20Distribution%20of%20Customers.png)

[Annual Income by Gender](images/Annual%20Income%20by%20Gender.png)

[Annual Income vs Spending Score](images/Annual%20Income%20vs%20Spending%20Score.png)

[Average Income and Spending Score by Customer Cluster](images/Average%20Income%20and%20Spending%20Score%20by%20Customer%20Cluster.png)

[Correlation Heatmap of Customer Features](images/Correlation%20Heatmap%20of%20Customer%20Features.png)

[Customer Age Distribution](images/Customer%20Age%20Distribution.png)

[Customer Distribution by Gender](images/Customer%20Distribution%20by%20Gender.png)

[Customer Feature Pairplot](images/Customer%20Feature%20Pairplot.png)

[Customer Segmentation Based on Income and Spending-2](images/Customer%20Segmentation%20Based%20on%20Income%20and%20Spending-2.png)

[Customer Segmentation Based on Income and Spending](images/Customer%20Segmentation%20Based%20on%20Income%20and%20Spending.png)

[Customer Segmentation using K-Means Clustering](images/Customer%20Segmentation%20using%20K-Means%20Clustering.png)

[Customer Segmentation with Cluster Centroids](images/Customer%20Segmentation%20with%20Cluster%20Centroids.png)

[Elbow Method for Optimal Number of Clusters](images/Elbow%20Method%20for%20Optimal%20Number%20of%20Clusters.png)

[Elbow Method](images/Elbow%20Method.png)

[Number of Customers in Each Segment](images/Number%20of%20Customers%20in%20Each%20Segment.png)

[Silhouette Score for Different Numbers of Clusters](images/Silhouette%20Score%20for%20Different%20Numbers%20of%20Clusters.png)

[Silhouette Score](images/Silhouette%20Score.png)

[Spending Score Distribution of Customers](images/Spending%20Score%20Distribution%20of%20Customers.png)

[Spending Score by Gender](images/Spending%20Score%20by%20Gender.png)

---

Author

Vadlamuri Sohan Aravind

Data Science | Machine Learning | Business Analytics

