# A Machine Learning Approach Using the Pima Indians Diabetes Dataset

This repository features a machine learning model trained on the Pima Indians Diabetes dataset, which contains various medical attributes of female patients. The aim of this project is to accurately predict the likelihood of diabetes using attributes such as glucose level, blood pressure, body mass index (BMI), age, and more. The model employs established machine learning techniques to analyze these features and provide insights into diabetes risk, making it a valuable tool for healthcare professionals and researchers in understanding diabetes prevalence.  

## Table of Contents  

- [Project Overview](#project-overview)  
- [Dataset Description](#dataset-description)  
- [Installation Instructions](#installation-instructions)  
- [Model and Techniques](#model-and-techniques)      
- [Results and Performance](#results-and-performance)  
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