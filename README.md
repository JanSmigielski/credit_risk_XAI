# To Explain the Unexplainable – Utilization of XAI in Credit Risk Modelling
This project explores how Explainable Artificial Intelligence (XAI) techniques can enhance the transparency of complex models used in credit risk assessment.

The primary goal is to compare a fully interpretable logistic regression model with a high-performing, but opaque, XGBoost classifier. By applying modern XAI methods—such as SHAP and LIME—to the XGBoost model, we aim to achieve an interpretability level similar to that of logistic regression, while benefiting from superior predictive power.

## Setting Up the Conda Environment

Follow these steps to create and activate a new Conda environment for this project:

### 1. Create a new environment (Python 3.10 recommended)

```bash
conda create -n credit-risk-xai python=3.10
```

### 2. Activate the enviroment

```bash
conda activate credit-risk-xai
```

### 3. Install the required packages from requirements.txt

```bash
pip install -r requirements.txt
```

