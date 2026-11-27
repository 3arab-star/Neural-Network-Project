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
