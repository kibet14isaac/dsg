# Diabetes Risk Prediction: Contributing to SDG 3 (Good Health and Well-being)

---

## Project Overview

This project utilizes machine learning to predict the risk of diabetes in individuals based on various health metrics. Developed as part of an assignment focusing on the **Sustainable Development Goals (SDGs)**, this solution directly contributes to **SDG 3: Good Health and Well-being**, by enabling early identification and proactive management of diabetes. Early detection can lead to timely interventions, lifestyle adjustments, and improved health outcomes for at-risk individuals.

## Problem Statement

Diabetes is a significant global health challenge, leading to severe complications if undiagnosed or untreated. The goal of this project is to develop a predictive model that can assess an individual's likelihood of developing diabetes, thereby supporting preventative healthcare initiatives and reducing the long-term impact of the disease.

## Machine Learning Approach

This project employs a **supervised learning (classification)** approach. The models are trained on a labeled dataset where the outcome (diabetic or non-diabetic) is known, allowing them to learn patterns and make predictions on new, unseen data.

## Dataset

The **Pima Indians Diabetes Database** is used for this project. This open-source dataset contains diagnostic measurements for female patients of Pima Indian heritage, including:

* `Pregnancies`: Number of times pregnant
* `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
* `BloodPressure`: Diastolic blood pressure (mm Hg)
* `SkinThickness`: Triceps skin fold thickness (mm)
* `Insulin`: 2-Hour serum insulin (mu U/ml)
* `BMI`: Body mass index (weight in kg/(height in m)^2)
* `DiabetesPedigreeFunction`: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
* `Age`: Age in years
* `Outcome`: Class variable (0 = non-diabetic, 1 = diabetic)

You can typically find this dataset on platforms like [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or the UCI Machine Learning Repository. Please ensure the `diabetes.csv` file is in the root directory of this project.

## Tools and Technologies

* **Python 3.x**
* **Jupyter Notebook:** For interactive development and visualization.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For machine learning model building, preprocessing (e.g., `StandardScaler`, `train_test_split`), and evaluation metrics.
* **Matplotlib & Seaborn:** For data visualization.

## Project Structure# dsg
