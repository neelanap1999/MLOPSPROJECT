# Credit Risk Assessment - MLOps

[Kush Suryavanshi](https://github.com/Kush210/)  | [Aryan Fernandes](https://github.com/aryanf12122000/) | [Naveen Pasala](https://github.com/------/)  |  [Shiva Naga Jyothi Cherukuri](https://github.com/ShivaNagaJyothi-Cherukuri/)  | [Neel Shirish Anap](https://github.com/neelanap1999/)  | [Vishesh Gupta](https://github.com/visheshgupta-BA/)


<p align="center">  
    <br>
	<a href="#">
	<img height=130 src="https://github.com/neelanap1999/MLOPSPROJECT/blob/main/Image/python.png" alt="Docker" title="Docker" hspace=20 />
        <img height=100 src="https://cdn.svgporn.com/logos/airflow-icon.svg" alt="Airflow" title="Airflow" hspace=20 /> 
        <img height=150 src="https://github.com/neelanap1999/MLOPSPROJECT/blob/main/Image/flow.svg" title="Tensorflow" hspace=20 /> 
        <img height=100 src="https://cdn.svgporn.com/logos/google-cloud.svg" alt="Google Cloud Platform" title="Google Cloud Platform" hspace=20 /> 
        <img height=130 src="https://github.com/neelanap1999/MLOPSPROJECT/blob/main/Image/Docker.png" alt="Flask" title="Flask" hspace=20 /> 
 
  </a>	
</p>
<br>


## 1. Introduction

Assessing credit risk is a critical task for any lending institution. That's why we've embarked on an ambitious project that combines the latest machine learning innovations with modern software engineering best practices. Our goal? To build a robust, scalable system capable of accurately predicting loan outcomes.

At the heart of our approach lies a powerful stack of tools and technologies. We're harnessing the flexibility of Python and the might of frameworks like TensorFlow to construct highly accurate predictive models. But we're not stopping there – we're employing cutting-edge techniques like transfer learning, ensemble methods, and neural architecture search to push the boundaries of model performance.

What truly sets our project apart, however, is its holistic nature. We've engineered an automated pipeline that streamlines the entire data science lifecycle, from data ingestion and preprocessing to model deployment and monitoring. With orchestration tools like Airflow and data processing engines like Google Cloud Dataflow, we've achieved unparalleled levels of automation, scalability, and reproducibility.

Embracing the principles of DevOps, we've adopted containerization with Docker, continuous integration and deployment with GitHub Actions, and rigorous testing with Tensorflow. Version control with Git and experiment tracking with MLflow foster an environment of collaboration, iteration, and knowledge-sharing among our team of data scientists and engineers.

Our ultimate goal? To deliver a best-in-class lending experience that empowers both borrowers and lenders. By combining machine learning innovation with software engineering excellence, we're poised to transform the credit risk assessment landscape, ensuring accurate decision-making, regulatory compliance, and a superior overall experience for all stakeholders.

## 2. Dataset Information

### 2.1. Dataset Introduction:
The dataset consists of 28 features for each applicant where loan_status is our target variable. The target variable has Fully_Paid and Charged_Off values indicating that the applicant either paid back the entire loan on time or missed all payments. Here Charged_Off is the class of interest as we want to identify potential risks in time.

### 2.2. Data Card:
The dataset has 396,030 entries where each entry corresponds to one applicant. There are 12 numeric features and 16 categorical features.

| Variable Name | Role | Type | Description |
|---------------|------|------|-------------|
| loan_amnt | Feature | Continuous | The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value. |
| term | Feature | Categorical | The number of payments on the loan. Values are in months and can be either 36 or 60. |
| int_rate | Feature | Continuous | Interest Rate on the loan |
| installment | Feature | Continuous | The monthly payment owed by the borrower if the loan originates. |
| grade | Feature | Categorical | LC assigned loan grade |
| sub_grade | Feature | Categorical | LC assigned loan subgrade |
| emp_title | Feature | Categorical | The job title supplied by the Borrower when applying for the loan. |
| emp_length | Feature | Ordinal | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. |
| home_ownership | Feature | Categorical | The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER |
| annual_inc | Feature | Continuous | The self-reported annual income provided by the borrower during registration. |
| verification_status | Feature | Categorical | Indicates if income was verified by LC, not verified, or if the income source was verified |
| issue_d | Feature | Date | The month which the loan was funded |
| loan_status | Feature | Categorical | Current status of the loan |
| purpose | Feature | Categorical | A category provided by the borrower for the loan request. |
| title | Feature | Categorical | The loan title provided by the borrower |
| zip_code | Feature | Categorical | The first 3 numbers of the zip code provided by the borrower in the loan application. |
| addr_state | Feature | Categorical | The state provided by the borrower in the loan application |
| dti | Feature | Continuous | A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower's self-reported monthly income. |
| earliest_cr_line | Feature | Date | The month the borrower's earliest reported credit line was opened |
| open_acc | Feature | Integer | The number of open credit lines in the borrower's credit file. |
| pub_rec | Feature | Integer | Number of derogatory public records |
| revol_bal | Feature | Continuous | Total credit revolving balance |
| revol_util | Feature | Continuous | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit. |
| total_acc | Feature | Integer | The total number of credit lines currently in the borrower's credit file |
| initial_list_status | Feature | Categorical | The initial listing status of the loan. Possible values are – W, F |
| application_type | Feature | Categorical | Indicates whether the loan is an individual application or a joint application with two co-borrowers |
| mort_acc | Feature | Integer | Number of mortgage accounts. |
| pub_rec_bankruptcies | Feature | Integer | Number of public record bankruptcies |


### 2.3. Data Source:
The dataset is a public dataset owned by [Lending Club](https://www.lendingclub.com/personal-savings/founder-savings)

Attached is the URL for data:
[Dataset Link](https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction/input?select=lending_club_loan_two.csv)


# Setup Instructions
Please confirm that `Python >= 3.8` or a later version is present on your system prior to installation. This software is designed to be compatible with Windows, Linux, and macOS platforms.

# Prerequisities
1. git
2. python>=3.8
3. docker daemon/desktop is running
4. apache-airflow==2.8.3
5. Flask==2.2.5

## Installation Steps for Users
To install for users, follow these steps:

1. Clone the repository to your local machine:
```
git clone https://github.com/Thomas-George-T/Ecommerce-Data-MLOps](https://github.com/neelanap1999/MLOPSPROJECT.git
```
2. Ensure that your Python version is 3.8 or above:
```python
python --version
```
3. Check if your system has sufficient memory:
```docker
docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
```
<hr>

## 3. Data Planning and Splits
Our dataset has 396k data points. We plan to use 100k for initial model development to get us metrics for the first iteration. The rest of the dataset will be grouped in batches of 10k and will be treated as new data for continuous training of our selected model.

### 3.1. Data Collection:
The initial phase of our machine learning project involves gathering pertinent data from trustworthy sources. This process may entail acquiring data from internal databases, external vendors, or openly accessible datasets. For our loan status prediction initiative, we will procure historical data on past loan applicants from either LendingClub's databases or other financial institutions. Ensuring the collected data is comprehensive, precise, and representative of the target audience is paramount.

### 3.2. Data Cleaning and Preprocessing:
Following data collection, we'll engage in data cleaning and preprocessing to ensure its integrity and suitability for analysis. This encompasses tasks like managing missing values, eliminating duplicates, handling outliers, and encoding categorical variables. For our project, we'll preprocess the loan applicant data to rectify inconsistencies or discrepancies, standardize feature formats, and ready the dataset for model training.

### 3.3. Exploratory Data Analysis (EDA):
EDA is a pivotal step in grasping the dataset's characteristics and interrelations. Through techniques such as visualizations, summary statistics, and hypothesis testing, we aim to discern variable distributions, detect patterns, correlations, and anomalies. EDA serves as a compass, guiding our feature selection and engineering endeavors by shedding light on the data's underlying structure.

### 3.4. Data Splitting:
Before embarking on model training, we will partition the dataset into distinct subsets: training, validation, and test sets. The training set will serve as the foundation for model training, the validation set for fine-tuning hyperparameters and selecting optimal models, and the test set for ultimate model evaluation. While a typical split ratio might be 70-15-15, adjustments may be made based on dataset size and project specifications.

### 3.5. Addressing Class Imbalance:
Should the dataset exhibit class imbalance, where one class (e.g., defaulted loans) is significantly underrepresented compared to another (e.g., non-defaulted loans), we'll employ various techniques such as oversampling, undersampling, or utilizing class weights to rectify this imbalance. This ensures that our machine learning models remain unbiased and capable of accurately predicting both classes.

### 3.6. Data Scaling and Normalization:
Depending on the algorithms utilized, we may need to standardize or normalize features to ensure they share a similar scale and distribution. This is particularly crucial for algorithms sensitive to feature scales or those reliant on distance metrics, such as support vector machines or k-nearest neighbors.

By thoughtfully strategizing and executing these steps within the data pipeline, we can ensure our machine learning models are trained on robust data, thereby enhancing the accuracy of our loan status predictions and deriving actionable insights.

## 4. GitHub Repository
- GitHub Link: [MLOPSPROJECT](https://github.com/neelanap1999/MLOPSPROJECT)
- README: [README](https://github.com/neelanap1999/MLOPSPROJECT/blob/main/README.md)

## 5. Project Scope

### 5.1. Problems:

1. Lack of Efficient Risk Assessment:
   - The current risk assessment processes may lack efficiency in identifying factors contributing to loan defaults, resulting in suboptimal risk management.

2. Limited Utilization of Data for Decision-Making:
   - The dataset contains valuable information about past loan applicants, and there is a need to harness this data for predictive modeling to enhance decision-making.

3. High Credit Loss Due to Defaults:
   - The largest source of financial loss for LendingClub is credit loss, particularly from loans extended to 'risky' applicants who eventually default. Identifying these risky applicants is crucial for minimizing credit loss.

## 5.2. Current Solutions:

1. **Manual Risk Assessment:**
   Traditional lending institutions often rely on manual risk assessment processes, where loan officers evaluate applicants based on a set of predetermined criteria. This approach involves subjective judgments and may not capture the nuanced relationships between various factors influencing loan default. Additionally, manual assessments can be time-consuming and prone to human biases, limiting their scalability and effectiveness in handling large volumes of loan applications.

2. **Rule-Based Decision Systems:**
   Some lending institutions employ rule-based decision systems that follow predefined guidelines to evaluate loan applications. These rules are typically based on historical data and institutional policies. While these systems provide a structured approach, they might lack adaptability to evolving market trends and changing borrower behaviors. The rigid nature of rule-based systems may lead to overlooking complex patterns and interactions among variables, hindering the accurate identification of risky applicants.

3. **Limited Use of Predictive Analytics:**
   Some financial institutions may use basic predictive models, but these often lack sophistication and may not leverage the full potential of available data. The limited incorporation of advanced analytics hampers the accurate assessment of loan default risk, and institutions may miss out on the opportunity to enhance decision-making through machine learning techniques.

Understanding the limitations of these current solutions highlights the opportunity to advance the loan status prediction process using more sophisticated and data-driven methodologies.

## 5.3. Proposed Solutions:

1. **Predictive modeling with Machine Learning:**
   We will leverage advanced machine learning algorithms to develop predictive models for loan status prediction. We will explore and implement algorithms such as logistic regression, decision trees, and ensemble methods like random forests and gradient boosting. The focus will be on creating a robust model that can effectively discover patterns in historical data, leading to accurate predictions of loan default.

2. **Feature Engineering for Risk Factors:**
   We will perform thorough feature engineering to identify and create relevant variables that contribute significantly to the prediction of loan default. This involves extracting insights from the existing dataset, transforming variables, and generating new features that capture the nuances of borrower behavior and financial attributes. The goal is to enhance the predictive power of the models by incorporating domain-specific knowledge and data-driven insights.

3. **Model Monitoring and Feedback Loop:**
   We will implement a robust model monitoring system to track the performance of deployed models in real-time. Establish a feedback loop that captures new data and user interactions, allowing for model recalibration and improvement over time. This iterative process ensures that the model remains effective in capturing evolving patterns in borrower behavior.

## 6. Current Approach Flow Chart and Bottleneck Detection

The current approach flowchart is given below to help visualize our steps. There are a few bottlenecks that can hinder the pipeline.

1. Engineering new features to monitor their impact on the target variable can consume time.
2. Imbalanced data and class skew can cause errors in metrics.

![Workflow](https://github.com/neelanap1999/MLOPSPROJECT/blob/main/Image/workflow.png)

## 7. Metrics, Objectives, and Business Goals

### 7.1 Primary Objectives:

The primary objective of this project is to develop a predictive model for loan status prediction, aimed at enhancing risk assessment processes within LendingClub. By leveraging machine learning techniques, the project seeks to identify key factors contributing to loan defaults and improve the efficiency of risk management strategies.

### 7.2 Model Performance Metrics:

To evaluate the effectiveness of the predictive model, we will utilize relevant performance metrics such as accuracy, precision, recall, and F1-score. These metrics will allow us to assess the model's ability to correctly classify loan applicants into default and non-default categories. Additionally, we will consider the area under the ROC curve (AUC-ROC) to evaluate the model's discriminatory power and overall predictive performance and models feature weights.

### 7.3 Business Goals:

1. **Enhance Risk Assessment Efficiency:** The project aims to improve the efficiency of risk assessment processes by identifying the most influential factors contributing to loan defaults. This will enable LendingClub to make more informed lending decisions and minimize the risk of financial loss due to defaults.

2. **Optimize Data Utilization:** By harnessing the valuable information contained within the dataset of past loan applicants, the project seeks to maximize the utilization of data for predictive modeling. This will enable LendingClub to leverage historical data to enhance decision-making and improve the accuracy of loan status predictions.

3. **Minimize Credit Loss:** The project aims to address the challenge of high credit loss resulting from loans extended to 'risky' applicants who default on their payments. By accurately identifying these risky applicants through predictive modeling, LendingClub can implement targeted risk mitigation strategies to minimize credit loss and improve overall financial performance.

## 8. Failure Analysis

1. **Data Quality Issues:** Poor data quality or class imbalance could lead to inaccurate model predictions and unreliable results. So, Proper data slicing techniques would be used to avoid the data quality issues in training, validation and testing sets.

2. **Model Performance:** The model fails to meet performance expectations, leading to inaccurate predictions.

3. **Infrastructure Issues:** Infrastructure failures or limitations impact the stability and scalability of the MLops pipeline.

4. **Deployment Challenges:** Difficulties in deploying models to production environments lead to delays or errors.

5. **Monitoring and Logging Issues:** Inadequate monitoring and logging make it difficult to detect and diagnose issues.

6. **Lack of Continuous Improvement:** Failure to incorporate feedback and iterate on the system leads to stagnation.

## 9. Deployment Infrastructure

1. Model Development will be done in Jupyter Notebook using frameworks like TensorFlow, or scikit-learn, and changes are tracked using Git for version control, ensuring collaboration and traceability in the development process.

2. We aim to use Docker for containerizing the application, along with its dependencies and trained model, facilitating consistency and portability across different environments.

3. For Model and Data Registry we would be using MLflow and DVC which employs to store and version trained models and data, seamlessly integrated with Git for efficient model and data management and versioning.

4. CI/CD Automation tools would be used to automate the testing, building, and deployment processes, streamlining development workflows and ensuring reliable and repeatable deployments.

5. Artifactory will be utilized to store Docker images, model artifacts, and other deployable components, promoting a centralized and organized approach to artifact management.

6. ELK stack (Elasticsearch, Logstash, Kibana) will be implemented for comprehensive logging, while Grafana will be used for monitoring tools, enabling the tracking of model performance and health over time.

7. Kubernetes will be employed for container orchestration, offering scalability and fault tolerance, crucial for efficiently managing and deploying containerized applications.

8. We would also like to use an API gateway to maintain and secure the API endpoints. Along with the tools to define and manage the infrastructure in a version-controlled manner.

9. GCP is used for cloud deployment that provides scalability, storage, and robust infrastructure management.

## 10. Monitoring Plan

The problem statement of the company is to identify various features that most contribute towards the loan status. In other terms, the company wants to figure out what are the causes of loan default so they can mitigate their financial risks. In the project, we plan to monitor the true positive rate, accuracy and weights of each feature in order to see how the importance of these features change over time.

1. **Define Metrics:** Determine which metrics are important to monitor based on project goals (accuracy, F1-score, feature weights).

2. **Data Monitoring:** Continuously monitor data quality, distribution shifts, and anomalies that could affect model performance.

3. **Model Monitoring:** Monitor model predictions, performance, and drift over time to detect degradation or changes in behavior.

4. **Feedback Loop:** Incorporate monitoring feedback into model retraining and improvement processes to maintain or enhance performance over time.

## 11. Success and Acceptance Criteria

1. The deployed machine learning model should achieve a satisfactory level of accuracy, precision, recall, and F1-score in predicting loan statuses.

2. The workflow automation implemented with Airflow should successfully orchestrate the entire ML pipeline, from data preprocessing to model deployment, without manual intervention.

3. The project should be well-documented, allowing team members to reproduce experiments, model training, and deployments accurately.

4. All code, models, datasets, and configurations should be version-controlled using Git, ensuring traceability and accountability.

5. Implement robust monitoring and logging mechanisms to track system performance, detect anomalies, and troubleshoot issues effectively.
   

## 12. Timeline Planning

1. **Data Exploration and Preprocessing (Week 1):**
   - Data loading and initial exploration.
   - Address outliers and anomalies in the dataset.
   - Conduct preliminary statistical analysis.

2. **Feature Engineering (Week 2):**
   - In-depth analysis of feature distributions.
   - Create new features based on domain knowledge.
   - Encode categorical variables for model compatibility.
   - Finalize feature selection for predictive modeling.

3. **Model Development (Weeks 3 - 4):**
   - Split the dataset into training and testing sets.
   - Choose and implement machine learning models for loan status prediction.
   - Fine-tune model hyperparameters for optimal performance.
   - Evaluate model performance using relevant metrics.

4. **Model Deployment (Week 5):**
   - Develop a deployment strategy for the machine learning model.
   - Implement the deployment process, ensuring compatibility with the chosen deployment environment.
   - Conduct thorough testing in the deployment environment to verify model functionality.

5. **Continuous Model Enhancement (Weeks 6-7):**
   - Implement a scalable and automated pipeline for model training, validation, and deployment.
   - Set up version control for models and integrate automated testing.
   - Establish protocols for continuous model updates based on new data.

6. **Model Monitoring and Feedback Loop (Week 8):**
   - Implement a real-time monitoring system to track model performance.
   - Establish a feedback loop capturing new data and user interactions.
   - Iteratively recalibrate the model for improved predictions.

7. **Documentation (Week 9):**
   - Document the entire model development process, including feature engineering, model selection, and deployment strategy.

8. **Buffer for Contingencies (Week 10):**
   - Allocate a buffer week for unexpected delays or additional tasks.
   - Address any last-minute adjustments or unforeseen challenges.


