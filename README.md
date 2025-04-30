Smoking and Drinking Dataset Analysis
Overview
This project analyzes the Smoking and Drinking Dataset with Body Signals from Kaggle. The dataset contains information about individuals' smoking and drinking habits along with various body signals. The goal is to explore the data, visualize key patterns, and build machine learning models to predict drinking behavior (DRK_YN).
Dataset

Source: Kaggle Smoking and Drinking Dataset
Description: The dataset includes features such as age, smoking status (SMK_stat_type_cd), drinking status (DRK_YN), and various physiological measurements (e.g., sight, hearing, blood pressure).
Preprocessing:
Handled 26 duplicate rows.
No missing values found.
Encoded categorical variables (SMK_stat_type_cd, DRK_YN) using LabelEncoder.
Scaled numerical features using StandardScaler.



Project Structure

Code: The main script (analysis.ipynb or equivalent Python file) contains:
Data loading and preprocessing.
Exploratory Data Analysis (EDA) with visualizations.
Machine learning model training and evaluation.


Dependencies: Listed in the Requirements section.
Outputs: Visualizations (bar plots, histograms, boxplots, heatmaps, etc.) and model performance metrics.

Analysis
Exploratory Data Analysis (EDA)

Visualizations:
Bar plots for SMK_stat_type_cd and DRK_YN distributions.
Pie chart for smoking status proportions.
Boxplots and histograms for numerical features.
Correlation heatmap for numerical variables.
Bar plots comparing age with sight and hearing metrics by drinking and smoking status.


Key Findings:
Balanced dataset for DRK_YN (Drinker vs. Not Drinker).
Identified highly correlated feature pairs using a correlation threshold of 0.6.
Smoking status: Majority are "Never" smokers, followed by "Current smoker" and "Used to smoke."



Machine Learning

Target Variable: DRK_YN (Drinker: 1, Not Drinker: 0).
Models Trained:
Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
(Decision Tree was commented out due to expected poor performance.)


Evaluation Metrics:
Accuracy, Precision, Recall, F1 Score.
Confusion matrices visualized for each model.


Data Split: 75% training, 25% testing.
Feature Scaling: Applied StandardScaler to numerical features.

Requirements
To run the code, install the following Python libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn

Install them using:
pip install pandas numpy matplotlib seaborn scikit-learn

How to Run

Download the Dataset:
Obtain the dataset from Kaggle.
Update the path variable in the script to point to the downloaded .zip file.


Install Dependencies:
Ensure all required libraries are installed (see Requirements).


Run the Script:
Execute the Python script or Jupyter notebook.
The script will:
Extract and load the dataset.
Perform EDA with visualizations.
Train and evaluate machine learning models.
Display results (plots and metrics).





Results

Model Performance: Detailed in the script output (classification reports and bar plots).
Visualizations: Saved as plots within the script (e.g., correlation heatmap, confusion matrices).
Highly Correlated Features: Identified pairs of features with correlation > 0.6.

Future Improvements

Experiment with additional models (e.g., XGBoost, Neural Networks).
Perform feature selection to reduce dimensionality.
Address potential class imbalance in SMK_stat_type_cd.
Hyperparameter tuning for better model performance.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset provided by Sooyoung Her.
Built using Python data science libraries: pandas, scikit-learn, matplotlib, seaborn.

