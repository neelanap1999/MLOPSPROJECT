# Data

## 1. Dataset Information:
The dataset consists of 28 features for each applicant where loan_status is our target variable. The target variable has Fully_Paid and Charged_Off values indicating that the applicant either paid back the entire loan on time or missed all payments. Here Charged_Off is the class of interest as we want to identify potential risks in time.

<p align="center">
<img src="Image/Class%20Distribution.png" alt="ML Project class" height="300">
</p>

## 2. Data Card:
The dataset has 396,030 entries where each entry corresponds to one applicant. There are 12 numeric features and 16 categorical features.

![Data Card](Image/Data%20Card.png)

## 3. Data Sources:
The dataset is a public dataset owned by [Lending Club](https://www.lendingclub.com/personal-savings/founder-savings)

Below is the URL for data:

[Dataset Link](https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction/input?select=lending_club_loan_two.csv)

We have structured our data pipeline into distinct modules, starting from data ingestion through preprocessing, ensuring that our data is well-prepared for modeling purposes. To ensure the functionality of each module, we adopt Test Driven Development (TDD) principles, rigorously testing every aspect of our pipeline.

Our data pipeline is orchestrated using Apache Airflow, where we design a Directed Acyclic Graph (DAG) comprising our modularized components. This approach streamlines the execution of our pipeline and enhances its scalability and reliability.

![Airflow Chart](Image/airflow_chart.png)

## 4. Airflow Setup:

In our data preprocessing workflow, Apache Airflow plays a critical role in automating and orchestrating tasks seamlessly. Leveraging Airflow's capabilities, we have designed a robust workflow automation system that manages Directed Acyclic Graphs (DAGs) efficiently.

The Airflow platform, built with Python, streamlines the execution of data preprocessing tasks, such as data ingestion, cleaning, transformation, and feature engineering. By defining tasks and their dependencies within Airflow's DAGs, we ensure that each step is executed in the correct sequence, optimizing the overall data preprocessing process.

Airflow's scheduler and web server components enable us to monitor and manage our workflows through a user-friendly web interface. This interface allows us to visualize the workflow, track task statuses, and troubleshoot any issues that may arise during execution. Below is the link to our code.

[Airflow Code Link](https://github.com/neelanap1999/MLOPSPROJECT/blob/main/dags/airflow.py)

## 5. Data Pipeline Components:

This project's data pipeline comprises multiple interconnected modules, each dedicated to executing specific data processing tasks. Our approach involves utilizing Airflow and Docker to orchestrate and encapsulate these modules within containers. Each module serves as an individual task within the primary data pipeline DAG (datapipeline), contributing to the seamless execution and management of our data processing workflows. The below chart explains that well. 

**Image of chart**

1. Preprocessing Data: During this stage, the dataset undergoes several cleaning and preprocessing procedures to guarantee data quality and prepare it for analysis. The subsequent modules participate in this operation.

- dataload.py - Script for loading and importing data into the pipeline.
- data_pipeline.py - Module responsible for orchestrating the data processing pipeline workflow.
- column_drop.py - Handles dropping unnecessary columns from the dataset during preprocessing.
- missing_values.py - Manages the handling of missing values through imputation or removal strategies.
- null_drop.py - Manages null values by dropping rows or columns containing null values.
- outlier_handle.py - Implements techniques to identify and handle outliers in the dataset to improve data quality.
    
2. Feature Engineering : During this phase, we engage in feature engineering to analyze and enhance the features, aiming to improve training outcomes and evaluation metrics. The subsequent modules are developed specifically for feature engineering purposes.

- correlation.py - Calculates and analyzes correlations between different features in the dataset.
- credit_year.py - Extracts credit years from timestamps and performs relevant data transformations.
- dummies.py - Generates dummy variables for categorical features using one-hot encoding.
- income_normalization.py - Normalizes income data to ensure consistency and comparability across different scales.
- pca.py - Applies Principal Component Analysis (PCA) to reduce dimensionality and extract essential features.
- scaling.py - Performs feature scaling to standardize numerical features and improve model performance.
- term_map.py - Maps and categorizes loan terms for better analysis and modeling.
- transform_emp_length.py - Transforms employment length data for better understanding and utilization in models.
- zipcode_extract.py - Extracts relevant information from zip codes for geographical analysis and modeling purposes.
