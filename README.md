# CyberThreat-LLM

### AI-Powered Network Threat Detection and Analysis

CyberThreat-LLM is a cybersecurity project focused on analyzing network traffic and detecting potentially malicious activity using machine learning.

The project combines network traffic preprocessing, Random Forest classification, model evaluation, explainability techniques, and risk analysis in an interactive Streamlit application.

It also includes support for analyzing external CSV files, allowing users to test the classification workflow with additional network traffic samples.

> **Project Status:** The current implementation focuses on machine learning-based threat detection and analysis. Large Language Model (LLM) integration is planned as a future improvement.

---

## 1. Project Overview

The main goal of CyberThreat-LLM is to explore how machine learning can support network security analysis and help identify suspicious network activity.

The project covers the following areas:

- Network traffic exploration and preprocessing
- Binary and multiclass classification experiments
- Random Forest model training and validation
- Analysis of benign and malicious traffic
- Model performance evaluation and visualization
- Feature importance and prediction explainability
- Risk-oriented threat analysis
- External CSV-based traffic classification
- Interactive visualization using Streamlit

The system is designed as an experimental cybersecurity analysis tool rather than a production-ready intrusion detection system.

---

## 2. Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core development |
| Pandas | Dataset exploration and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Model training and evaluation |
| Random Forest | Network traffic classification |
| SHAP | Model explainability |
| Streamlit | Interactive web interface |
| Joblib | Model serialization and loading |
| Matplotlib | Data and model visualizations |

---

## 3. Datasets

The project uses publicly available network intrusion detection datasets containing statistical features extracted from network traffic.

### CICIDS2017

CICIDS2017 is used for data exploration, preprocessing, model development, and network attack classification experiments.

The dataset contains both benign traffic and several categories of malicious activity.

The project includes scripts for preparing data, training classification models, and evaluating predictions.

### CICIDS2018

CICIDS2018 is included in the workflow for additional dataset preparation and external evaluation experiments.

Using a separate dataset helps investigate how a trained model behaves when it encounters traffic outside its original training distribution.

### External CSV Data

The application also supports working with external CSV files containing network traffic records.

A sample input file is included in the repository:

`sample_traffic.csv`

This file can be used to explore the external data analysis workflow.

External CSV files must follow the feature structure expected by the trained model. Column names, data types, and preprocessing requirements should be compatible with the model's input format.

---

## 4. System Architecture

The general workflow of the project is:

```text
                 Network Traffic Data
                          |
                          v
                    CSV Loading
                          |
                          v
                  Data Exploration
                          |
                          v
                    Preprocessing
                          |
                          v
                 Feature Preparation
                          |
                          v
                Random Forest Model
                          |
                          v
                 Threat Classification
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Model Explainability       Risk Analysis
              |                       |
              +-----------+-----------+
                          |
                          v
                  Streamlit Dashboard
                          |
                          v
               Visualizations & Reports
```

The system processes network flow features, generates classification predictions, and provides additional analysis to support the interpretation of detected threats.

---

## 5. Machine Learning Models

The project includes both baseline and multiclass classification experiments.

### Random Forest Classification

Random Forest is used to identify patterns in network traffic and classify records according to the classes represented in the training data.

The repository includes scripts for:

- Baseline model training
- Multiclass model training
- Model validation
- Prediction analysis
- External dataset preparation
- Threat prediction
- Risk analysis

### Model Evaluation

The classification models can be evaluated using common machine learning metrics:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall proportion of correct predictions |
| Precision | Reliability of positive or class-specific predictions |
| Recall | Ability to identify actual instances of a class |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Distribution of correct and incorrect predictions |

These metrics are particularly important for intrusion detection datasets, where benign traffic may significantly outnumber certain attack categories.

Evaluation results should be interpreted alongside class distributions and the potential impact of false positives and false negatives.

---

## 6. Data Visualizations and Analysis

An important part of this project is examining both the network traffic data and the behavior of the trained models.

### Dataset Exploration

The analysis scripts are used to investigate the structure and characteristics of network traffic datasets.

This includes examining:

- Dataset dimensions and feature columns
- Traffic class distributions
- Missing and invalid values
- Feature characteristics
- Differences between benign and malicious traffic

### Model Performance Analysis

Model evaluation helps identify classification strengths and weaknesses.

Performance metrics and confusion matrices can be used to investigate how effectively the model distinguishes different traffic categories.

### Feature Importance

Feature importance analysis helps identify which network flow characteristics contribute most strongly to the model's predictions.

This provides additional context for understanding how the classification model makes decisions.

### Prediction Explainability

The project includes prediction explanation functionality using SHAP-related techniques.

These explanations help investigate how individual features influence a model's output.

Model explanations describe the behavior of the classifier; they do not independently prove that a network event is malicious.

### Risk Analysis

The risk analysis component provides a security-oriented interpretation of classification results.

It is intended to help users review suspicious traffic and understand why further investigation may be necessary.

---

## 7. Interactive Streamlit Dashboard

The project includes a Streamlit-based interface that brings together network traffic analysis, machine learning predictions, and visualizations.

The dashboard is intended to make the analysis workflow easier to explore without requiring users to execute each Python script separately.

Depending on the available input data and model configuration, the application can be used to review:

- Network traffic classification results
- Data and prediction visualizations
- Model analysis outputs
- Risk-related information
- External CSV analysis results

### Development and AI Assistance

I used ChatGPT as a development assistant while building the Streamlit interface, particularly for UI implementation, layout organization, and troubleshooting.

My main focus throughout this project has been on data preprocessing, machine learning, threat detection, and model analysis.

Since I am still improving my skills in application and interface development, AI assistance helped me organize and connect these components into a usable dashboard.

---

## 8. External CSV Analysis

One of the practical features of CyberThreat-LLM is the ability to analyze additional network traffic samples using CSV files.

This allows the classification workflow to be explored with data beyond the original training dataset.

### Example CSV File

The repository includes:

```text
sample_traffic.csv
```

The sample file provides a reference for preparing compatible external input data.

### How to Use External CSV Data

**Step 1 — Prepare the CSV file**

Prepare a CSV file containing network flow records.

The file should include the features required by the trained model.

**Step 2 — Start the Streamlit application**

Run:

```bash
streamlit run app.py
```

**Step 3 — Load the CSV file**

Use the CSV input functionality in the application to select the provided sample file or another compatible CSV file.

**Step 4 — Run the classification**

The application processes the input data and uses the trained model to generate traffic classification predictions.

**Step 5 — Review the analysis**

Examine the available prediction results, visualizations, and risk analysis information.

### CSV Compatibility

External data should satisfy the following requirements:

- Feature names must be compatible with the model.
- Input values should use the expected numerical formats.
- Missing or invalid values may require preprocessing.
- The feature set should match the model's training configuration.
- Previously unseen traffic may produce less reliable predictions.

The provided sample CSV is intended as a starting point for testing the workflow.

---

## 9. Project Structure

```text
CyberThreat-LLM/
|
|-- app.py
|-- requirements.txt
|-- sample_traffic.csv
|-- README.md
|
|-- src/
|   |-- analyze_all_datasets.py
|   |-- analyze_model.py
|   |-- check_multiclass.py
|   |-- explain_prediction.py
|   |-- explore_data.py
|   |-- predict_threat.py
|   |-- prepare_external_2018.py
|   |-- prepare_multiclass.py
|   |-- preprocess.py
|   |-- risk_analysis.py
|   |-- train_baseline.py
|   |-- train_multiclass.py
|   `-- validate_multiclass.py
|
|-- data/
|   `-- external/
|
`-- models/
    |-- bruteforce_random_forest.joblib
    |-- multiclass_random_forest.joblib
    `-- multiclass_random_forest_backup.joblib
```

**Note:** The `models/` directory represents the local project structure. Trained model files may need to be generated or added separately if they are not available in the public repository.

Large raw and processed datasets may also require separate local setup.

---

## 10. Installation and Usage

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sudeilhn/CyberThreat-LLM.git
cd CyberThreat-LLM
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Prepare the Required Models

Make sure the trained model files required by `app.py` are available in the expected directories.

If the model files are not included in the repository, prepare the necessary datasets and generate the models using the corresponding training scripts.

### Step 5 — Launch the Application

```bash
streamlit run app.py
```

The application typically opens in a browser at:

`http://localhost:8501`

### Step 6 — Explore the Dashboard

Use the available application features to inspect network traffic predictions, analysis results, and visualizations.

For external CSV testing, prepare a compatible input file and use the application's supported CSV analysis workflow.

---

## 11. Limitations

The project is still under development and has several limitations:

- Model performance depends on the quality and distribution of the training data.
- Models may not generalize equally well to previously unseen datasets.
- False positives and false negatives are possible.
- External CSV files require compatible network flow features.
- Large datasets and trained models may require additional setup.
- The current implementation does not include a fully integrated LLM-based analysis pipeline.
- The system is not intended for production security monitoring without further validation.

---

## 12. Future Improvements

Planned improvements include:

- Integrating an LLM for natural-language threat explanations
- Improving model interpretability and security reporting
- Expanding evaluation across additional intrusion detection datasets
- Comparing different machine learning algorithms
- Improving CSV input validation and compatibility
- Enhancing the Streamlit interface
- Exploring real-time network traffic processing

---

## 13. Disclaimer

This project is intended for educational, research, and defensive cybersecurity purposes.

The classification results, model explanations, and risk assessments should be treated as supporting information rather than definitive evidence of malicious activity.
