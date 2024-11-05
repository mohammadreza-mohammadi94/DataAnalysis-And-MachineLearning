#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

"""# Download Dataset"""

# !kaggle datasets download -d ashfakyeafi/spam-email-classification

# !unzip /content/spam-email-classification.zip

"""## Import Dataset"""

df = pd.read_csv("/content/email.csv")
df

"""# Convert Categories to Binary Format"""

df['Is_Spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()

"""# Split Dependent and Independent Variables and Train/Test Sets"""

X = df['Message']
y = df['Is_Spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""# Naive Bayes Model"""

# Create pipeline for Naive Bayes
nv_pipeline = Pipeline(
    [
        ('vectorizer', TfidfVectorizer()),
        ('nb', MultinomialNB())
    ]
)

# Train naive bayes model
nv_pipeline.fit(X_train, y_train)

# Predict on test set
y_pred_nv = nv_pipeline.predict(X_test)

"""## Confusion Matrix"""

cnf_matrix_nv = confusion_matrix(y_test, y_pred_nv)

sns.heatmap(cnf_matrix_nv, cmap='Blues', annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')

print(classification_report(y_test, y_pred_nv))

"""# Train Naive Bayes Model By SMOTE Algorithm"""

# Convert text to TF-IDF features
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)


smt = SMOTE(random_state=42,sampling_strategy=0.5)
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

print(X_train_sm.shape)
print(y_train_sm.shape)

y_train_sm.value_counts()

nb = MultinomialNB()
nb.fit(X_train_sm, y_train_sm)
y_pred_sm = nb.predict(X_test)

"""## Confusion Matrix"""

cnf_nb_sm = confusion_matrix(y_test, y_pred_sm)

sns.heatmap(cnf_nb_sm, cmap='Blues', annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')

print(classification_report(y_test, y_pred_sm))

