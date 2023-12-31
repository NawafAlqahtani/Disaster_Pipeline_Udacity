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

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Instructions:

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database **python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db**
To run ML pipeline that trains classifier and saves python **models/train_classifier.py data/DisasterResponse.db models/classifier.pkl**
Run the following command in the app's directory to run your web app. **python run.py**

Go to **http://0.0.0.0:3001/**
