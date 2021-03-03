# AG-News-Classification-using-Machine-Learning
News Classification Dataset
Data Source:https://www.kaggle.com/amananandrai/ag-news-classification-dataset/notebooks

# Objective 

News Classification dataset consist of News articles of class "world", "sports", "business", and "Science"

Given a Title and description we have to determine wheatear it belongs to which news article category .

## Type of Machine learning problem:

We have to predict the news article on given information so it is multiclass Classification problem




## Basic Overview

Data.shape : Train.csv + Test.csv = 120000 + 7600 =127600 rows.

Data.columns : Class index , Title , Description

Data.info( ) : Independent : Title , Description --- > Object  , Dependent : Class Label -- >  Int64

## Type of Machine Learning problem

![image](https://user-images.githubusercontent.com/61958476/109811522-f683e900-7c50-11eb-8293-bc2a0e345cee.png)


## Performance metric 

As this is Multiclass Classification problem so we are going to use:

1: Multiclass Confusion matrix

2: Precision , Recall ,F1-Score

3: Accuracy score , Error score

## Steps for doing News Classification problem:

![image](https://user-images.githubusercontent.com/61958476/109811703-321eb300-7c51-11eb-9768-f76a01d44097.png)

### Part 1

1: Load dataset ---- > .csv format

2: Perform Exploratory Data Analysis: 

      a] Check if Dataset has balanced distributions for each News label 
          b] Check for null values 
          c] Plot Distribution of data points among News Labels .
          d] Use word clouds to observe max repeated words in each class.

### Part 2 Text Preprocessing

Pre-processing : 

1: Expand Contradictions --->  (.replace (‘ % ‘ , percent), .replace ( ‘ $ ‘ , dollar’ )

2: Remove Html tags , links ,url.

3 : Remove Punctuations .

4: Remove Stop words   ---->( is , the , are …..)

5: Perform Stemming operations to convert more than one words with different spelling  with similar meaning into one meaningful word

### Part 3 Train Test Validation Split

Divide Dataset into 3 parts Dtrain , Dtest, Dval into 60 , 20 , 20 ratio

Dtrain : max amount of data is used to training this data for model to learn from it

Dval : after training Dtrain our model we have to validate our data to see that our model have learned in proper manner or not.

Dtest : Unseen data

### Part 4 Apply NLP model

For creating a model : 

1: Tf-ifd

2: Uni-gram, bi-gram, n-gram

3: Selecting max_features out of the model

### TF-IDF

#### Here i have created a small blog explaining TF-IDF intiution

https://niharjamdar.medium.com/tf-idf-term-frequency-and-inverse-document-frequency-56a0289d2fb6

![image](https://user-images.githubusercontent.com/61958476/109812230-d43e9b00-7c51-11eb-87c7-4c6602c4c242.png)

### Why use log in IDF

![image](https://user-images.githubusercontent.com/61958476/109812281-e4ef1100-7c51-11eb-93ed-50abc2a2e98f.png)

![image](https://user-images.githubusercontent.com/61958476/109812305-eae4f200-7c51-11eb-828f-5e51ffaef13c.png)


### Part 5 Apply ML models

After Applying NLP models for creating words into vectors .

We will convert those words into features by selecting  max_features , max_df  as hyper parameter.

More the features more machine will learn , and after that use machine learning algorithm:

1 : Logistic regression

2: Decision Tree

3: Stochastic Gradient descent



