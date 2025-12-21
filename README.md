# ML-CapstoneProject
ML PROJECT AI3


Machine Learning Capstone Project

Heart Disease Prediction & Clustering

📌 Project Overview

This project is a comprehensive Machine Learning Capstone Project that demonstrates the complete machine learning pipeline using a real-world medical dataset.

The project covers:

Data loading and cleaning

Exploratory Data Analysis (EDA)

Feature preprocessing

Classification (Heart Disease Prediction)

Clustering (K-Means)

Model evaluation and visualization

Professional documentation and reproducibility

The goal is to predict the presence of heart disease and explore hidden patterns in patient data using clustering techniques.

📊 Dataset Information

Dataset Name: Heart Disease Prediction Dataset
Source: Kaggle

Type: Medical / Tabular data

Rows: Patients

Columns: Clinical attributes (age, sex, chest pain, cholesterol, etc.)

Target Variable:

target

1 → Heart disease present

0 → No heart disease

📁 Dataset location:

data/heart.csv

🗂 Project Structure
ML-CapstoneProject/
│
├── data/
│   └── heart.csv
│
├── notebooks/
│   ├── 03_eda.ipynb
│   └── ML CODE.ipynb
│
├── results/
│   ├── Elbow Method.jpg
│   ├── K-means Clusters.jpg
│   ├── 3D K-means Clusters Visualization.jpg
│   ├── Outlier Detection.jpg
│   ├── actual vs predicted.jpg
│   └── confusion Matrix.jpg
│
├── project_report.pdf
├── .gitignore
└── README.md

⚙️ Environment Setup
1️⃣ Clone the Repository
git clone https://github.com/mohamed0-ai/ML-CapstoneProject.git
cd ML-CapstoneProject

2️⃣ Create & Activate Virtual Environment
python -m venv venv


Windows:

.\venv\Scripts\activate

3️⃣ Install Required Libraries
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

▶️ How to Run the Project
1️⃣ Launch Jupyter Notebook
jupyter notebook

2️⃣ Run Notebooks (in order)
🔹 03_eda.ipynb

Data inspection

Statistical analysis

Outlier detection

Feature distributions

Correlation analysis

🔹 ML CODE.ipynb

Data preprocessing

Feature scaling

Model training

Evaluation

Visualization

Saving figures to results/

🤖 Models Implemented
🔹 Classification Task

Objective: Predict the presence of heart disease

Model: Logistic Regression

Metrics:

Accuracy

Confusion Matrix

Precision

Recall

📊 Output visualization:

Confusion Matrix

Actual vs Predicted comparison

🔹 Clustering Task

Objective: Identify hidden patient groups

Model: K-Means Clustering

Techniques:

Elbow Method (optimal K)

Silhouette analysis

2D & 3D cluster visualization

📊 Output visualization:

Elbow Method plot

Cluster scatter plots

🖼 Results & Visualizations

All generated figures are stored in:

results/


These include:

Classification evaluation plots

Clustering visualizations

Outlier detection graphs

All figures are used directly in the final report.


📄 project_report.pdf

The report includes:

Introduction & problem definition

Dataset description

EDA findings

Classification results

Clustering analysis

Visualizations

Conclusion & future work

Google Colab link 

🧪 Tools & Technologies

Python

Pandas & NumPy

Scikit-learn

Matplotlib & Seaborn

Jupyter Notebook

Git & GitHub

🎯 Conclusion

This project demonstrates how machine learning can be applied to medical data to support decision-making and pattern discovery while following ethical, reproducible, and professional ML practices.

👤 Author

Mohamed Hassan
GitHub: https://github.com/mohamed0-ai
