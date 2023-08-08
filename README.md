## Description
In this project, I'll apply data engineering to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

data directory contains a data set which are real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that appropriate disaster relief agency can be reached out for help.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Project Overview

There are three components of this project:

### 1.ETL Pipeline
File process_data.py contains data cleaning pipeline that:

1.Loads the messages and categories dataset
2.Merges the two datasets
3.Cleans the data
4.Stores it in a SQLite database

### 2.ML Pipeline
File ML Pipeline Preparation.py contains machine learning pipeline that:

1.Loads data from the SQLite database
2.Splits the data into training and testing sets
3.Builds a text processing and machine learning pipeline
4.Trains and tunes a model using GridSearchCV
5.Outputs result on the test set
6.Exports the final model as a pickle file.


### 2.3 Flask Web App

Running this command from app directory will start the web app where users can enter their query, i.e., a request message sent during a natural disaster.


  

