# Machine Learning Assignment 2: Dry Bean Classification

## a. Problem statement

In this assignment, we are building and comparing different machine learning models to classify dry beans into seven distinct types. The main goal is to see which algorithm does the best job of predicting a bean's variety based solely on its physical measurements (like its area, perimeter, and shape).

## b. Dataset description

I chose the **Dry Bean Dataset** from the UCI Machine Learning Repository for this project.

* **Instances:** 13,611 rows of data.

* **Features:** 16 continuous numeric features. These are basically physical dimensions extracted from images of the beans (e.g., MajorAxisLength, MinorAxisLength, AspectRatio).

* **Target:** 1 categorical column (`Class`) containing the 7 bean varieties (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira).

This dataset fits the assignment constraints perfectly (well over 500 instances and 12+ features). It's also fully numeric with no missing values, making it a great clean dataset to test out these baseline models. Full dataset file is present in the dataset directory of this project.

## c. Github Repository Link

https://github.com/2026aj05295-source/2026aj05295_ml_assignment2

## d. Models used:

I trained and evaluated the following five machine learning models on a 20% test split:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | 0.9214 | 0.9948 | 0.9222 | 0.9214 | 0.9216 | 0.9050 | 
| **Decision Tree** | 0.8920 | 0.9450 | 0.8917 | 0.8920 | 0.8917 | 0.8695 | 
| **KNN** | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 | 
| **Naive Bayes** | 0.8979 | 0.9916 | 0.9007 | 0.8979 | 0.8981 | 0.8773 | 
| **Support Vector Machine** | 0.9221 | 0.9910 | 0.9224 | 0.9221 | 0.9222 | 0.9059 | 

### Observations on Model Performance

| ML Model Name | Observation about model performance | 
| ----- | ----- | 
| **Logistic Regression** | Did surprisingly well, actually getting the highest AUC score of the group (0.9948). The physical traits of the beans must be pretty linearly separable. | 
| **Decision Tree** | This one had the lowest scores overall. It probably overfit a bit on the training data and didn't generalize as well to the test set as the other models. | 
| **KNN** | Solid accuracy (0.9166). Since the dataset is mostly physical dimensions, grouping beans by how "close" they are in distance makes a lot of sense. | 
| **Naive Bayes** | Had a great AUC score, but the accuracy was a bit lower. It assumes all features are independent, which isn't really true here (e.g., a bean's area and perimeter are obviously related). | 
| **Support Vector Machine** | Just slightly edged out Logistic Regression to take the top spot in almost every main metric. | 
| **Overall, Winner for dataset?** | **Support Vector Machine (SVM)**. It gave the best Accuracy (0.9221) and MCC (0.9059). It seems to handle the overlapping physical features slightly better than the rest of the algorithms. | 

### Streamlit App link

https://2026aj05295mlassignment2-csp3nr5zs7weus9qs27e4q.streamlit.app
