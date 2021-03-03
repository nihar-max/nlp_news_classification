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

Part 1

1: Load dataset ---- > .csv format

2: Perform Exploratory Data Analysis: 

      a] Check if Dataset has balanced distributions for each News label 
          b] Check for null values 
          c] Plot Distribution of data points among News Labels .
          d] Use word clouds to observe max repeated words in each class.

