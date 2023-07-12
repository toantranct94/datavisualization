# Diabetes-Prediction

Table of Content
-------------------
* Demo
* Directory Tree
* Overview
* Motivation
* Installation
* Advantages
* Steps
* Deployement on Heroku
* Result/Summary
* Future scope of project

Demo
-------
Link: https://diabetes-predictor-webapp.herokuapp.com/


![Diabetes1](https://user-images.githubusercontent.com/41515202/95026746-3601d580-06b1-11eb-9caf-7ec0657a5f42.PNG)

![Diabetes2](https://user-images.githubusercontent.com/41515202/95026747-37330280-06b1-11eb-859b-c043f5159d83.png)

![Diabetes3](https://user-images.githubusercontent.com/41515202/95026749-37330280-06b1-11eb-888d-bf7bd2bc45bd.png)




Directory Tree
-----------------

├── static

│ ├── css

├── template

│ ├── home.html

├── Procfile

├── README.md

├── app.py

├── diabetes_model.pkl

├── requirements.txt


Overview / What is it ??
-------------------------
* This is a simple Flask web app which predicts whether a patient is having diabetes or not.

* Diabetes is a chronic condition in which the body develops a resistance to insulin, a hormone which converts food into glucose & affect many people worldwide and is normally divided into Type 1 and Type 2 diabetes & considered as one of the deadliest and chronic diseases which causes an increase in blood sugar.

Motivation / Why / Reason ??
-------------------------------
* What to do when you are at home due to this pandemic situation? I started to learn Machine Learning model to get most out of it. I came to know mathematics behind all unsupervised models. Finally it is important to work on application (real world application) to actually make a difference.

* But the rise in machine learning approaches solves this critical problem. The motive of this study is to design a model which can prognosticate the likelihood of diabetes in patients with maximum accuracy.

* Analyzed & explored dataset , perform EDA and create a model to predict if a particular observation is at a risk of developing diabetes, given the independent factors & performances of all the three algorithms are evaluated on various measures like Precision, Accuracy, F-Measure, and Recall. Accuracy is measured over correctly and incorrectly classified instances.


Installation / Tech Used
------------------------
The Code is written in Python 3.6.10. If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository

pip install -r requirements.txt

Used two datasets, Train data and Test data from Kaggle

Language – Python, Anaconda

Other libraries for analyzing & visualization: Pandas, Numpy, Matplotlib, Seaborn

AI/ML : Scikit-Learn , ML models

Web Frameworks : Flask

Hosting: Heroku (side projects & demos)

Tracking & SC: GitHub


![gunicorn](https://user-images.githubusercontent.com/41515202/95001323-62045480-05e6-11eb-8002-8df9dbf65adf.png)
![pyton](https://user-images.githubusercontent.com/41515202/95001324-629ceb00-05e6-11eb-87c9-c2bceb4cac7e.png)
![sklearn](https://user-images.githubusercontent.com/41515202/95001325-63358180-05e6-11eb-9653-89e2f8a3cc37.png)
![Anaconda](https://user-images.githubusercontent.com/41515202/95001326-63ce1800-05e6-11eb-9b4c-53cc5d267a2f.png)
![flask](https://user-images.githubusercontent.com/41515202/95001327-6466ae80-05e6-11eb-89ed-8b828e7ac949.png)

 
 
Advantages
------------
* help to show early diseases of diabetic

* Glucose Monitoring Systems: Machine learning algorithms recommend optimal insulin dosages to maintain balanced glucose levels

* Machine learning algorithms help automate the process of monitoring blood sugar levels and recommend adjustments in care.

* Nutrition Coaching: To help recommend meal options based on the specific diet criteria of the user.


Phases - Timeline
-------------------
1). Data Collection - Importing Dataset

2). EDA - Feature Engeneering( Dividing data into features and labels), Selection, Explore dataset, Data Cleaning, Convert categorical data into numerical, Concatenate both catagorical and numerical data

3 ). ML Models Selection - Building Supervised Machine Learning Models => Xgboost, Random Forest, KNN, Gradientboost

4). Evaluation - Used in a 10-fold cross-validation procedure to train the aforementioned ML models. The performance indices used to compare the models are the prediction 
accuracy (% - MSE between the desired and predicted prices) and the time in seconds, needed to train each model.

5). Deployment - Deployed on Heroku using Flask framework


Process
------------
* Since data is in form of excel file we have to use pandas read_excel to load the data

* We will be using train and test data. We can do some data pre-processing and remove variables which are not needed

* After loading it is important to check the complete information of data as it can indication many of the hidden infomation such as null values in a column or a row. Next step is Feature generation, here we mainly work on the data and do some transformations to extract unknown variables or create different bons of particular columns and clean the messy data.

* Check whether any null values are there or not. if it is present then following can be done:
    Imputing data using Imputation method in sklearn
	   Filling NaN values with mean, median and mode using fillna() method
    Describe data, which can give statistical analysis

* Mainly work on the data set and do some transformation like creating different bins of particular columns ,clean the messy data so that it can be used in our ML model . This step is very important because for a high prediction score you need to continuously make changes in it

* Do some EDA, analysis & data visualisation to understand the relationship between different independent variables and the relationship between the independent variables and the dependent variables

* Prepare categorical variables for model using label encoder - convert categorical text data into model-understandable numerical data

* Divide the data set into test and train - all our data is numerical after label encoding so we split the data into test and train & predict the price with our test data set

* Building Model - measure the performance of a better and more tuned algorithm, & using different Classifier Technique and comparing them to see which algorithm is giving better performance. Evaluated various models for computing expected future prices and classifying whether this is the best time to buy the ticket. Finally after the above steps. Predict the air tickets prices, and the performance of the models is compared to each other. Later deployed the model and evaluate the efficiency of the predictions.

* UNSUPERVISED TMODELS USED: Random Forest Classifier: 90.04% KNN : 75.7% Xgboost : 87.48% Gradientboost : 87.59% ACCURACY SCORE : 93:14%

Deployement on Heroku
-------------------------
* Login or signup in order to create virtual app. You can either connect your github profile or download ctl to manually deploy this project.
 
 ![Heroku](https://user-images.githubusercontent.com/41515202/95001317-4ac56700-05e6-11eb-9450-98107d78461d.png)
 
* Our next step would be to follow the instruction given on Heroku Documentation to deploy a web app.


Result / Summary
-------------------
* Developed end-to-end full fleged ML-WebApp to display price accurate predictions with Random Forest 87% accuracy & deployd on heroku

* Applied all various high performance ML models & compared their best f1, recall, precision, ROC, support score to predict the diabetes

* Evaluated 4 unsupervised classifier models Xgboost, Random Forest, KNN, Gradientboost & improved & optimized the Model by HPT Cross validation, GridSearchCV

* Demonstrated EDA, handling categorical data, feature selection & scaling, dimensionality reduction & feature transformation using PCA Bias-Variance tradeoff, Performance Metrics, Splitting data into train and test set, evaluated random forest classifier confusion matrix(y_test,y_pred), performed GRidSearchCV to optimize model,cross validated k-fold, trained model using Random Forest Regressor

* Recommended Random Forest Classifier, with presicion = 83% , recall = 83% , f1score = 83% , support = 154,ROC score = 0.81% accuracy = 90.04% on test data, model HP Tuning using Cross validation, mean = 0.7167451503781488 , SD = 0.05709939039060164, GRidSearchCV with best accuracy score = 0.755700325732899


Future Scope
-------------------
Use multiple Algorithms.
Optimize Flask app.py.
Front-End.

