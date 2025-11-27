# Elliptic Dataset - Data Mining Project

## Project Overview

This project aims to build a comprehensive preprocessing and analysis pipeline on the Elliptic dataset, a publicly available dataset of Bitcoin transactions. The primary goal is to explore the dataset thoroughly through data preprocessing, exploratory data analysis (EDA), feature engineering, unsupervised learning, and supervised learning techniques. This work lays the foundation for future graph mining exploration on the Elliptic dataset.

The project is implemented using Jupyter Notebooks and Python libraries such as pandas, scikit-learn, matplotlib, seaborn, and XGBoost.

---

## Dataset

The Elliptic dataset contains transaction data from the Bitcoin network, including features extracted from transactions and labels indicating whether a transaction is illicit, licit, or unknown.

- **Source:** [Elliptic Bitcoin Dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- **Files used:**
  - `elliptic_txs_features.csv`: Transaction features
  - `elliptic_txs_classes.csv`: Transaction labels (classes)

---

## Project Structure & Workflow

### 1. Data Fusion and Preparation
- Download and merge transaction features and labels.
- Save the merged dataset for subsequent analysis.

### 2. Exploratory Data Analysis (EDA)
- Overview of dataset dimensions and types.
- Checking for duplicates and missing values.
- Visualizing class distribution.
- Histograms, boxplots, and heatmaps to understand feature distributions and correlations.
- Outlier detection using Z-score and visualization of correlations among features.

### 3. Feature Engineering
- Detect and treat outliers (univariate and multivariate) using IQR and Isolation Forest.
- Feature selection:
  - Remove near-constant features based on variance threshold.
  - Remove highly correlated features (correlation > 0.9).
  - Use Mutual Information for supervised feature relevance.
- Data normalization and standardization for algorithm compatibility.
- Save cleaned and processed datasets for modeling.

### 4. Unsupervised Learning
- Apply K-Means clustering to identify inherent transaction groups.
- Use Elbow Method and Silhouette Score to select the optimal number of clusters.
- Perform clustering on both full and PCA-reduced datasets.
- Visualize clusters in 2D PCA space and analyze outlier distribution within clusters.

### 5. Supervised Learning
- Filter dataset to exclude 'unknown' class transactions.
- Further feature reduction using variance, correlation, and mutual information.
- Address class imbalance with SMOTE oversampling.
- Train and evaluate several classification models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - ExtraTreesClassifier
  - XGBoost
- Metrics used for evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve.
- Visualize feature importance and model performance.

---

## Goals and Future Directions

- **Goal:** Establish a robust preprocessing pipeline for the Elliptic dataset to enable effective data mining and machine learning tasks.
- **Future Work:** Explore advanced graph mining techniques on the Bitcoin transaction network to detect illicit activities and understand transaction patterns.

---

## Technologies and Libraries

- Python 3.x
- Jupyter Notebooks
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- imbalanced-learn (SMOTE)

---

## Usage

To reproduce the results:

1. Download the dataset from Kaggle or use the provided script to fetch it automatically.
2. Run the notebooks sequentially:
   - `fusion_csv.ipynb` for data merging
   - `EDA.ipynb` for exploratory analysis
   - `feature_engineering.ipynb` for preprocessing and feature selection
   - `unsupervised.ipynb` for clustering
   - `supervised.ipynb` for classification modeling

---

## Acknowledgements

- Dataset provided by Elliptic via Kaggle.
- Open-source Python libraries used for analysis and modeling.

---

