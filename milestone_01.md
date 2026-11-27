# 01. Problem Definition
### Bank Marketing Subscription Prediction

> **Project Domain:** Banking & Financial Services  
> **Machine Learning Type:** Supervised Learning  
> **Learning Task:** Binary Classification  
> **Dataset:** UCI Bank Marketing Dataset  
> **Primary Framework:** TensorFlow / Keras

---

## 1.1 Problem Statement

Banks conduct marketing campaigns to promote term deposit subscriptions. However, contacting every customer is inefficient, costly, and may result in low conversion rates.

This project aims to develop a predictive system that helps bank marketing and campaign management teams identify customers who are more likely to subscribe to a term deposit. Using customer demographic, financial, and campaign-related information, the system will predict whether a client will subscribe (`yes`) or not (`no`).

The prediction system will help improve customer targeting and optimize marketing resources. Incorrect predictions have practical consequences: False Positives may lead to unnecessary marketing expenses and customer contacts, while False Negatives may cause the bank to miss potential subscription opportunities.

The problem will be addressed using supervised learning through a Neural Network-based Binary Classification model.

## 1.2 Business Objectives

The proposed system focuses on the following objectives:

- **Improve Customer Targeting:** Identify customers with a higher likelihood of subscribing to a term deposit.
- **Optimize Marketing Resources:** Reduce unnecessary contact attempts and campaign expenses.
- **Support Decision-Making:** Provide prediction probabilities to help marketing teams prioritize potential customers.
- **Increase Campaign Efficiency:** Improve the effectiveness of customer outreach strategies.

## 1.3 Intended Users

| User / Stakeholder | Responsibility |
|:---|:---|
| Bank Marketing Teams | Identify and prioritize potential subscribers. |
| Campaign Management Teams | Improve campaign targeting and resource allocation. |
| Customer Relationship Management (CRM) Teams | Support customer engagement and outreach decisions. |

## 1.4 Prediction Target

The model will predict the likelihood of a customer subscribing to a bank term deposit.

| Component | Description |
|:---|:---|
| Target Variable | `y` |
| Positive Class | `yes` – Customer subscribed. |
| Negative Class | `no` – Customer did not subscribe. |
| Prediction Type | Binary Classification |
| Expected Output | Predicted class (`yes` / `no`) and subscription probability. |

## 1.5 Input Features

The model will use customer information grouped into three main categories:

| Feature Category | Examples |
|:---|:---|
| **Demographic Features** | Age, job, marital status, education. |
| **Financial Features** | Account balance, credit default, housing loan, personal loan. |
| **Campaign-Related Features** | Contact type, campaign month, number of contacts, previous campaign outcomes. |

*Note: Final feature selection and preprocessing decisions will be documented during the Exploratory Data Analysis (EDA) stage. Features unavailable at the intended prediction time will be excluded.*

## 1.6 Consequences of Incorrect Predictions

Two types of prediction errors are particularly important for this business problem:

| Error Type | Definition | Business Impact |
|:---|:---|:---|
| **False Positive (FP)** | Predicting `yes` when the actual outcome is `no`. | Wasted marketing resources and unnecessary customer contacts. |
| **False Negative (FN)** | Predicting `no` when the actual outcome is `yes`. | Missed potential customers and lost subscription opportunities. |

Understanding these errors will guide the selection and justification of the primary evaluation metric.

## 1.7 Machine Learning Formulation

| Component | Specification |
|:---|:---|
| Learning Paradigm | Supervised Learning |
| Problem Type | Binary Classification |
| Input | Customer demographic, financial, and campaign-related features. |
| Output | Two possible classes: `yes` or `no`. |
| Model Architecture | Multi-Layer Perceptron (MLP) |
| Output Activation | Sigmoid |
| Loss Function | Binary Cross-Entropy |
| Framework | TensorFlow / Keras |

The proposed MLP will learn patterns from historical customer records to estimate the probability of term deposit subscription.

---
# 02. Dataset Selection & Validation

### UCI Bank Marketing Dataset

> **Dataset Source:** UCI Machine Learning Repository  
> **Dataset ID:** 222  
> **Domain:** Banking & Financial Services  
> **Learning Task:** Binary Classification  
> **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 2.1 Dataset Overview

The selected dataset for this project is the **UCI Bank Marketing Dataset**, which contains information collected from direct marketing campaigns conducted by a Portuguese banking institution.

The dataset focuses on predicting whether a bank client will subscribe to a term deposit based on demographic, financial, and campaign-related information.

It provides a suitable foundation for developing a Neural Network-based Binary Classification system because it contains a relatively large number of customer records, mixed feature types, categorical unknown values, and an imbalanced target variable.

### Dataset Specifications

| Property | Description |
|:---|:---|
| Dataset Name | Bank Marketing |
| Repository | UCI Machine Learning Repository |
| Dataset ID | 222 |
| Total Records | 45,211 |
| Input Features | 16 |
| Total Columns | 17 (16 inputs + 1 target) |
| Target Variable | `y` |
| Target Classes | `yes`, `no` |
| Problem Type | Binary Classification |
| Feature Types | Numerical and Categorical |
| License | CC BY 4.0 |

**Official Dataset Source:**  
https://archive.ics.uci.edu/dataset/222/bank+marketing

**Direct Dataset Download:**  
https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

---

## 2.2 Dataset Requirements Verification

The selected dataset was evaluated against the minimum dataset requirements defined in the AIE231 Course Project Specification.

### A. Minimum Dataset Size

**Requirement:** At least 10,000 rows.

**Dataset Evidence:**
- The dataset contains 45,211 customer records.
- Each record represents a bank client associated with the marketing campaign data.

**Validation:**

45,211 > 10,000

**Result:** PASS

### B. Minimum Number of Features

**Requirement:** At least 10 features.

**Dataset Evidence:**
- The dataset contains 16 input features.
- These features describe customer demographics, financial information, and marketing campaign history.
- The target variable `y` is stored separately from the input features.

**Validation:**

16 > 10

**Result:** PASS

### C. Binary Classification & Class Imbalance

**Requirement:** Binary Classification with clear class imbalance.

**Dataset Evidence:**

The target variable `y` indicates whether a customer subscribed to a term deposit.

| Target Class | Meaning | Number of Records | Percentage |
|:---|:---|---:|---:|
| `no` | Did not subscribe | 39,922 | 88.30% |
| `yes` | Subscribed | 5,289 | 11.70% |
| **Total** | | **45,211** | **100%** |

The target distribution demonstrates the following class imbalance:

- `no`: 88.3%
- `yes`: 11.7%

**Validation:**
- The target contains exactly two classes.
- The positive class represents approximately 11.7% of the records.
- The dataset exhibits a clear imbalance between the two classes.

**Result:** PASS

### D. Real-World Data Imperfections

**Requirement:** The dataset must contain real-world imperfections, such as unknown values and mixed feature types.

**Dataset Evidence:**

The dataset contains different types of features and categorical values that require preprocessing.

**1. Mixed Feature Types**

| Feature Type | Examples |
|:---|:---|
| Numerical | `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous` |
| Categorical | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` |
| Target | `y` |

**2. Unknown Categorical Values**

Some categorical features contain an `unknown` category, including:

- `job`
- `education`
- `contact`
- `poutcome`

These values represent unavailable or unspecified information and will require appropriate handling during preprocessing.

**3. Special Values**

The dataset contains special values such as `-1` in the `pdays` feature, indicating that the client was not previously contacted in an earlier campaign.

These values require careful interpretation rather than being treated automatically as ordinary numerical measurements.

**4. Class Imbalance**

The target distribution is heavily concentrated in the `no` class, which introduces an additional challenge during model training and evaluation.

**Validation:**
- Mixed numerical and categorical features are present.
- Unknown categorical entries are present.
- Special values requiring preprocessing are present.
- The target distribution is imbalanced.

**Result:** PASS

### E. Public Availability & Academic Usage License

**Requirement:** The dataset must be publicly available with a license permitting academic use.

**Dataset Evidence:**
- The dataset is publicly accessible through the UCI Machine Learning Repository.
- It is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
- This license permits sharing and adaptation, provided appropriate attribution is given.

**Dataset Citation:**

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository.

https://doi.org/10.24432/C5K306

**Validation:**
- Publicly accessible: Yes.
- License documented: Yes.
- Academic use permitted: Yes, with attribution.

**Result:** PASS

### F. Dataset Source & Download Documentation

**Requirement:** Dataset source and download link must be documented.

| Resource | Link |
|:---|:---|
| Official UCI Dataset Page | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Direct Download | https://archive.ics.uci.edu/static/public/222/bank+marketing.zip |
| Dataset DOI | https://doi.org/10.24432/C5K306 |
| License Information | https://creativecommons.org/licenses/by/4.0/ |

**Validation:**
- Official repository identified.
- Direct download link provided.
- Dataset DOI documented.
- License reference included.

**Result:** PASS

### G. Computational Feasibility

**Requirement:** Dataset must be suitable for training within the project's computational limitations (less than 30 minutes on Free Google Colab GPU).

**Dataset Evidence:**
- The dataset contains 45,211 records and 16 input features.
- It is a relatively small tabular dataset compared with typical image and sequence datasets.
- The proposed MLP architectures are expected to have manageable computational requirements.

**Computational Considerations:**
- Categorical encoding and numerical scaling will be applied during preprocessing.
- Training will use mini-batches to manage memory consumption.
- The proposed MLP models will be evaluated on Google Colab.
- Training time will be measured and documented for the selected architectures.

**Validation:**
- Dataset size is manageable for tabular Neural Network experiments.
- The proposed model architecture is computationally appropriate.
- Actual compliance with the 30-minute training limit will be confirmed through an initial training benchmark.

**Result:** EXPECTED PASS — Pending empirical training-time verification.

---

