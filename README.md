# A Machine Learning Approach Using the Pima Indians Diabetes Dataset

This repository features a machine learning model trained on the Pima Indians Diabetes dataset, which contains various medical attributes of female patients. The aim of this project is to accurately predict the likelihood of diabetes using attributes such as glucose level, blood pressure, body mass index (BMI), age, and more. The model employs established machine learning techniques to analyze these features and provide insights into diabetes risk, making it a valuable tool for healthcare professionals and researchers in understanding diabetes prevalence.  

## Table of Contents  

- [Project Overview](#project-overview)  
- [Dataset Description](#dataset-description)  
- [Installation Instructions](#installation-instructions)  
- [Model and Techniques](#model-and-techniques)      
- [Results and Performance](#results-and-performance)
- [Improvements](#improvements)  
    - [KNN Imputation Method for Missing Values](#knn-imputation-method-for-missing-values)
    - [PCA Method for Feature Selection](#pca-method-for-feature-selection) 
    - [New Validation Methods](#new-validation-methods) 
- [Usage Instructions](#usage-instructions)  
- [Future Work](#future-work)

## Project Overview

This project aims to build a predictive model for diabetes using the Pima Indians Diabetes dataset. The model helps in identifying individuals at risk of diabetes based on medical attributes such as age, BMI, insulin levels, and more.

## Dataset Description

The Pima Indians Diabetes dataset is sourced from the National Institute of Diabetes and Digestive and Kidney Diseases. It includes data on 768 female patients of Pima Indian heritage, with the following attributes:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0 or 1, indicating the absence or presence of diabetes)

## Installation Instructions  
To run this project, ensure you have Python installed along with Jupyter Notebook. You'll also need the following libraries:  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- imblearn

To install these libraries, you can use pip:  
```bash  
pip install pandas numpy scikit-learn matplotlib seaborn imblearn
```

## Model and Techniques  

This project utilizes a Naive Bayes classifier to predict diabetes. Key steps include:  
- Data pre-processing: Handling missing values, detecting outliers, data normalization and balancing, feature selection to identify the most significant attributes, and splitting the data into training and testing sets.  
- Model training: Using a Naive Bayes classifier to train the model on the training set.  
- Model evaluation: Assessing the model's performance using accuracy, precision, recall, F1-score, classification report, confusion matrix, and AUC score.  

## Results and Performance

The model achieved an accuracy of 78% on the test set. Key performance metrics include:
- Precision: 0.63
- Recall: 0.56
- F1-score: 0.59
- AUC score: 0.71

Detailed performance metrics and visualizations are available in the results section of the repository.

## Improvements  

In this project, several improvements have been made to enhance the data preprocessing and model performance processes. 

1. **KNN Imputation for Missing Values.**
2. **PCA Method for Feature Selection.**
3. **New Validation Methods** 

### KNN Imputation Method for Missing Values  

In this update, I have added a new method for handling missing values using KNN Imputation. This method improves the data preprocessing step by providing a more robust way to impute missing values based on the nearest neighbors. The benefits of KNN Imputation are:  
- **Preservation of Relationships**: KNN considers the similarity of data points, which helps maintain the relationships between features when imputing missing values.  
- **Improved Model Performance**: Proper handling of missing data can lead to enhanced model accuracy and reliability.

### PCA Method for Feature Selection

A key aspect of this analysis is feature selection, which in this update, I employ Principal Component Analysis (PCA) to reduce the number of variables and retain the most relevant information.

Principal Component Analysis (PCA) is a dimensionality reduction technique. It transforms the original variables into a new set of uncorrelated variables called principal components, ordered by the amount of variance they capture from the data. By using PCA, we can efficiently eliminate less important features, enhancing model interpretability and performance.

### New Validation Methods

The model has been updated to incorporate new validation methods for enhanced accuracy and reliability, utilizing cross-validation and leave-one-out cross-validation techniques.

### 1. Cross-Validation  

Cross-validation is a statistical method used to estimate the skill of machine learning models. It involves partitioning the data into subsets, training the model on some subsets while validating it on others. This process is repeated several times to improve the accuracy of the model evaluation. The most common form is k-fold cross-validation, where the dataset is divided into 'k' subsets, and the model is trained and validated 'k' times, each time using a different subset as the validation set.

### 2. Leave-One-Out Cross-Validation (LOOCV)  

Leave-One-Out Cross-Validation is a special case of cross-validation where the number of subsets equals the number of data points in the dataset. This means that for each iteration, all data points except one are used for training, and the left-out data point is used for validation. This method provides a thorough evaluation of the model but can be computationally expensive for large datasets.

### Outcome  

The following are the results of various validation and performance metrics for the model:  

- **Holdout:** 0.7767  
- **Repeated random sampling:** 0.7767  
- **Cross-validation:** 0.7794  
- **Leave-one-out cross-validation:** 0.7794  
- **Naive Bayes testing:** 0.7813    

These outcomes indicate that the model's performance improved with the implementation of the updated validation techniques. 

## Usage Instructions

To use this project, clone the repository using the following command in your terminal or command prompt:

1. Clone the repository:
   ```bash
   git clone https://github.com/Ehsan-Behzadi/A-Machine-Learning-Approach-Using-the-Pima-Indians-Diabetes-Dataset.git  
   cd A-Machine-Learning-Approach-Using-the-Pima-Indians-Diabetes-Dataset
   ```
Next, open the Jupyter Notebook file (usually with a .ipynb extension) using Jupyter Notebook.   

2. To start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Future Work

Future improvements and directions for this project include:
- Exploring other classification algorithms such as Random Forest, K-Nearest Neighbors and more.
- Hyperparameter tuning to optimize model performance.
- Incorporating additional features to enhance prediction accuracy.