 Customer Segmentation Using K-Means Clustering

An end-to-end unsupervised machine learning project that applies K-Means clustering to segment mall customers based on their income and spending behavior.

---

Project Overview

This project applies Unsupervised Machine Learning to identify meaningful groups of mall customers based primarily on their Annual Income and Spending Score.

The objective is to transform raw customer data into meaningful customer segments that can support data-driven business strategies, including targeted marketing campaigns, personalized promotions, customer engagement, customer retention, cross-selling, upselling, and premium customer strategies.

The project demonstrates a complete Data Science workflow, from raw data analysis and exploratory analysis to machine learning, business interpretation, and interactive deployment using Streamlit.

---

🚀 Live Demo

[Customer Segmentation Streamlit App](https://customer-segmentation-kmeans-s2ewbtaskf9aryzluogpmd.streamlit.app/)

---

Business Problem

Businesses serve customers with different income levels, spending behaviors, and purchasing characteristics. Therefore, a single marketing strategy may not be equally effective for every customer.

Customer segmentation helps businesses identify groups of customers with similar characteristics and develop more relevant strategies for each group.

This project uses K-Means Clustering to identify customer groups based on income and spending behavior and transform the resulting clusters into meaningful business insights.

---

Project Objectives

The main objective of this project is to analyze mall customer data and identify meaningful customer segments using K-Means Clustering.

The project includes data quality analysis, exploratory data analysis, feature selection, feature scaling, cluster evaluation, customer segmentation, cluster profiling, business interpretation, business recommendations, results export, and Streamlit deployment.

The project also demonstrates how Machine Learning results can be transformed into meaningful business insights and practical customer engagement strategies.

---

Dataset

Dataset: Mall Customer Segmentation Data

Source: Kaggle

Records: 200 customers

Dataset Features

CustomerID: Unique identifier for each customer.

Gender: Customer gender.

Age: Customer age.

Annual Income (k$): Customer annual income.

Spending Score (1-100): Customer spending behavior score.

The primary features used for the final clustering analysis are Annual Income (k$) and Spending Score (1-100).

These features help identify customers with similar financial capacity and spending behavior.

---

Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

K-Means Clustering

Google Colab

Streamlit

GitHub

---

Project Structure

customer-segmentation-kmeans/

data/

Mall_Customers.csv

images/

Project visualizations and charts

Results/

Customer_Segmentation_Results.csv

Customer_Segmentation_Project.ipynb

README.md

app.py

requirements.txt

The data folder contains the original customer dataset.

The images folder contains exploratory data analysis, clustering, and model evaluation visualizations.

The Results folder contains the final customer segmentation results.

Customer_Segmentation_Project.ipynb contains the complete end-to-end Data Science and Machine Learning implementation.

app.py contains the Streamlit application for interactive customer segmentation.

requirements.txt contains the required Python libraries for running the project.

README.md contains the complete project documentation.

---

Project Workflow

The project follows a complete end-to-end workflow.

Data Loading

Data Understanding

Data Quality Analysis

Exploratory Data Analysis

Feature Selection

Feature Scaling

Elbow Method

Silhouette Score

K-Means Clustering

Cluster Profiling

Customer Segment Analysis

Business Recommendations

Model Evaluation

Results Export

Streamlit Deployment

---

Machine Learning Approach

Algorithm

K-Means Clustering

K-Means is an unsupervised machine learning algorithm used to group similar data points into clusters.

In this project, customers are grouped based on similarities in their Annual Income and Spending Score.

---

Selected Features

The final clustering analysis primarily uses Annual Income (k$) and Spending Score (1-100).

These features provide meaningful information for identifying customer groups based on financial capacity and spending behavior.

---

Feature Scaling

Feature scaling is applied before clustering because K-Means is a distance-based algorithm.

Standardizing the selected features helps ensure that differences in feature scales do not disproportionately influence the clustering process.

---

Cluster Selection

The number of clusters is evaluated using the Elbow Method and Silhouette Score.

Elbow Method

The Elbow Method analyzes the Within-Cluster Sum of Squares (WCSS) for different numbers of clusters.

Silhouette Score

The Silhouette Score evaluates how well-separated and cohesive the identified clusters are.

These evaluation methods help assess the clustering solution before finalizing the customer segments.

---

Key Analysis

The project performs exploratory and clustering analysis to understand customer demographics, customer age distribution, gender distribution, annual income distribution, spending score distribution, relationships between customer features, income and spending behavior, customer cluster characteristics, and segment size and distribution.

The analysis helps transform raw customer information into meaningful patterns and business-oriented customer groups.

---

Customer Segmentation

The K-Means clustering model assigns customers to groups based on similarities in income and spending behavior.

The numerical cluster assignments are then analyzed and interpreted as meaningful customer segments.

The overall analytical process can be summarized as:

Raw Customer Data

↓

Feature Analysis

↓

K-Means Clustering

↓

Cluster Assignment

↓

Customer Segment Interpretation

↓

Business Insights

---

Business Insights

The identified customer segments can support targeted marketing campaigns, personalized offers, improved customer engagement, customer retention strategies, cross-selling opportunities, upselling opportunities, and premium customer strategies.

Customer segmentation allows businesses to better understand different customer groups instead of applying the same strategy to every customer.

---

Business Recommendations

Business recommendations are developed according to the income and spending characteristics of each customer segment.

High-Value Customers

High-value customers can be targeted with personalized premium campaigns, loyalty rewards, exclusive offers, and long-term customer retention strategies.

High-Income Low-Spending Customers

High-income customers with lower spending behavior can be targeted using personalized product recommendations, engagement strategies, and incentives designed to encourage increased spending.

High-Spending Customers

High-spending customers can receive personalized promotions and relevant offers while businesses can explore opportunities for cross-selling and upselling.

Low-Spending Customer Groups

Low-spending customer groups can be targeted with suitable promotional campaigns, affordable products, and engagement strategies designed to encourage increased spending.

---

Model Evaluation

The clustering model is evaluated using Within-Cluster Sum of Squares (WCSS), the Elbow Method, Silhouette Score, cluster-size analysis, cluster visualization, cluster profiling, and business interpretability.

Since this is an unsupervised learning project, traditional classification metrics such as accuracy are not the primary evaluation criteria.

The quality of the model is evaluated based on cluster separation, cohesion, interpretability, and business usefulness.

---

Project Results

The final clustering results are exported as:

Customer_Segmentation_Results.csv

The results contain the original customer information along with numerical cluster assignments and customer segment labels.

This file represents the final analytical output of the customer segmentation process.

---

Streamlit Application

The project includes an interactive Streamlit application that allows users to enter customer income information and customer spending score information.

The application predicts the customer's cluster and displays the corresponding customer segment.

Users can also visualize customer clusters, view cluster centroids, analyze customer segment summaries, and explore business recommendations.

The Streamlit application demonstrates how the Machine Learning workflow can be transformed into an interactive and user-friendly analytical application.

---

Installation and Usage

Clone the repository.

git clone <repository-url>

Navigate to the project directory.

cd customer-segmentation-kmeans

Install the required libraries.

pip install -r requirements.txt

Run the Streamlit application.

streamlit run app.py

---

Key Learning Outcomes

This project demonstrates practical experience in Python programming, data analysis, data quality analysis, Exploratory Data Analysis, data visualization, feature selection, feature scaling, unsupervised machine learning, K-Means Clustering, the Elbow Method, Silhouette Score, cluster profiling, customer segmentation, business analytics, Streamlit deployment, and GitHub project documentation.

---

Project Outcome

This project demonstrates an end-to-end Data Science workflow that transforms raw customer data into meaningful customer groups and actionable business insights.

The complete workflow can be summarized as:

Raw Customer Data

↓

Data Quality Analysis

↓

Exploratory Data Analysis

↓

Feature Selection

↓

Feature Scaling

↓

Cluster Evaluation

↓

K-Means Clustering

↓

Cluster Profiling

↓

Business Segmentation

↓

Business Recommendations

↓

Results Export

↓

Streamlit Deployment

The project demonstrates both the technical implementation of an unsupervised machine learning model and the ability to translate analytical results into meaningful business insights.

---

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

