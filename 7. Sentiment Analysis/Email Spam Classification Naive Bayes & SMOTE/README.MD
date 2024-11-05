# Email Spam Classification

This project implements a machine learning model to classify emails as spam or not spam. The dataset used includes email messages with labels indicating if they are spam. The model leverages Natural Language Processing (NLP) techniques with Naive Bayes classifiers and addresses class imbalance with SMOTE.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Author](#author)

---

## Overview
This project uses a Naive Bayes classifier along with TF-IDF text vectorization to predict whether an email is spam. Additionally, the SMOTE algorithm is applied to balance the dataset and improve model performance on the minority class. The steps in this project include data preprocessing, training, evaluation, and visualization of results.

## Dataset
The dataset used in this project was downloaded from Kaggle and consists of labeled email messages. The categories are converted into binary labels:
- `spam` emails are labeled as `1`
- `not spam` emails are labeled as `0`

To download the dataset, run:
```bash
kaggle datasets download -d ashfakyeafi/spam-email-classification
unzip spam-email-classification.zip
```

## Project Structure
- **Data Loading and Preparation:** The dataset is loaded and labels are converted into binary format.
- **Train-Test Split:** The data is split into training and test sets.
- **Naive Bayes Classifier:** The initial model is trained using a Naive Bayes classifier with TF-IDF features.
- **SMOTE Resampling:** To address data imbalance, SMOTE is applied, and the model is retrained and evaluated.
- **Evaluation and Visualization:** Confusion matrices and classification reports are generated to evaluate model performance.

## Dependencies
This project uses the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imblearn`
- `warnings`

## Usage
1. **Clone the repository**

2. **Run the code**:
   The project code is implemented in a Jupyter Notebook format. You can open and run the code in a notebook environment, or convert it to a `.py` script if needed.

3. **Training and Evaluation**:
   - The Naive Bayes model is trained on the dataset with and without SMOTE resampling.
   - After training, evaluate the model by viewing the confusion matrix and classification report.

## Results
### Model Evaluation without SMOTE
The model is first trained without any data balancing. Results are displayed in terms of accuracy, precision, recall, and F1-score.

### Model Evaluation with SMOTE
Using SMOTE, the dataset is resampled, and the model is retrained. The balanced dataset improves performance on the minority class, achieving a more robust classification of spam emails.

Confusion matrices and classification reports are generated to visualize performance improvements after SMOTE application.


