#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:



# import libraries


import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet'])


# In[2]:


# load data from database
engine = create_engine('sqlite:///Df.db')

df = pd.read_sql_table('Df', engine)
X = df['message']
Y = df.iloc[:, 4:]


# In[3]:


df.head()


# In[4]:


X.head()


# In[5]:


Y.head()


# ### 2. Write a tokenization function to process your text data

# In[6]:


def tokenize(text):
    
    """
    Tokenization Function for NLP
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[8]:



pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, Y)


# In[10]:


pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[11]:


y_pred = pipeline.predict(X_test)


# In[12]:


def Model_Test(y_test, y_pred):
    
    """
    Model Testing on Each Column then it print the result of the model
    """
    for index, column in enumerate(y_test):
        print(column, classification_report(y_test[column], y_pred[:, index]))


# In[13]:


Model_Test(y_test, y_pred)


# In[14]:


accuracy = (y_pred == y_test).mean()
accuracy


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[15]:


pipeline.get_params()


# In[18]:


parameters = {
        'clf__estimator__min_samples_split': [2, 4]
    }

cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)


# In[ ]:


cv.fit(X_train, y_train)


# In[ ]:


cv.best_params_


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


y_pred = cv.predict(X_test)

Model_Test(y_test, y_pred)


# In[ ]:


accuracy = (y_pred == y_test).mean()
accuracy


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:


import pickle
filename = 'model.pkl'
pickle.dump(cv, open(filename, 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




