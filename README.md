# Credit Risk Model Project

This repository contains a series of notebooks and code for building a credit risk model. The project is organized into four main folders, each containing specific tasks to process the data, analyze variables, develop traditional models, and implement a credit risk model using machine learning.

---

## Folders Structure

### 1. **Data Cleaning**
This folder contains the data files and a Jupyter notebook for cleaning the dataset. The main objective of this folder is to clean the data and prepare it for further analysis. The data file `train_cleaned.csv` is generated after cleaning the `train` dataset. The notebook performs necessary preprocessing steps such as handling missing values, removing outliers, and ensuring the dataset is ready for modeling.

---

### 2. **Variable Analysis**
This folder contains a Jupyter notebook that performs an analysis of the variables in the `train_cleaned.csv` dataset. The notebook includes the following:
- **Heatmap**: Correlation heatmap to explore the relationships between numeric variables.
- **Violin Plots**: Plots to analyze the distribution of categorical variables with respect to the `Credit_Score` (Poor, Standard, Good).
- **Histograms**: Histograms for numeric variables showing their distributions and how they relate to the `Credit_Score` values. This analysis helps to identify which variables are relevant for building the scoring and credit risk model.

---

### 3. **Traditional Model**
This folder contains MATLAB code for building a traditional credit scoring model using the `train_cleaned.csv` dataset. The process includes:
- Using `creditscorecard`, `autobinning`, `fitmodel`, `formatpoints`, and `displaypoints` to create two traditional models:
  1. **Model for differentiating Poor and Standard scores**
  2. **Model for differentiating Standard and Good scores**
- The logic for assigning scores:
    - **Step 1**: Assign `scores_poor_std` values for the Poor vs Standard model below 600 points.
    - **Step 2**: Assign values from the second model (`scores_std_good`) to the remaining rows.
    - The final scores are stored and used to predict the categories (`Poor`, `Standard`, `Good`) based on the calculated scores. With the next classification rules:
      - Points < 600 = 'Poor'
      - 600 <= Points < 725 = 'Standard'
      - Points > 725 = 'Good'

Additionally, a Python implementation is included, which mirrors the principles and points seen in the MATLAB code.

#### RSAM Model Scoring Class

The `RSAMScoring` class is designed to calculate and classify credit scores based on predefined ranges for the next features:

`cols = {'Credit_History_Age_Months', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Outstanding_Debt', 'Delay_from_due_date', 'Num_Credit_Inquiries', 'Score_Category'};`

It also provides visualizations to analyze the distributions of different variables and the classification results.

##### Class Methods

 `__init__(self, df)`
Initializes the RSAMScoring class.  
**Parameters:**  
- `df`: pandas DataFrame containing required columns:  
  - `Credit_History_Age_Months`  
  - `Num_Bank_Accounts`  
  - `Num_of_Loan`  
  - `Outstanding_Debt`  
  - `Delay_from_due_date`  
  - `Num_Credit_Inquiries`  
  - `Score_Category` (Actual score category: Poor, Standard, Good)


###### `get_score(value, ranges)`
*Helper method*  
Returns the score for a given value based on specified ranges.  
**Parameters:**  
- `value`: Value to be scored  
- `ranges`: Predefined scoring ranges  
**Used internally** for value-to-score mapping.


###### `calculate_poor_std_points()`
Calculates points for "Poor vs Standard" classification using predefined feature ranges.  
**Adds column:**  
- `poor_std_points_total`: Total points per row for this classification


###### `calculate_std_good_points()`
Calculates points for "Standard vs Good" classification using predefined ranges.  
**Adds column:**  
- `std_good_points_total`: Total points per row for this classification


###### `calculate_final_points()`
Calculates final credit score by combining points from both classifications.  
**Adds column:**  
- `final_points`: Composite score for each row


###### `plot_histograms(variable)`
Generates a histogram showing variable distribution by credit score category.  
**Parameters:**  
- `variable`: Column name to visualize (must exist in DataFrame)  
**Output:** Density distribution plot for each score category.


###### `classify_final_points()`
1. Classifies `final_points` into categories (Poor/Standard/Good)  
2. Generates accuracy metrics and confusion matrix  
**Adds column:**  
- `predicted_category`: Final classification for each row  
**Output:** Model performance evaluation

---

### 4. **Credit Risk Model**
This folder contains a Jupyter notebook that implements a credit risk model using three machine learning algorithms:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

The notebook evaluates the performance of each model by calculating the **accuracy** and **AUC score**. It also generates **ROC curves** and **confusion matrices** for each model to assess their predictive power. The goal of this notebook is to implement machine learning techniques to predict credit risk and compare the performance of different models.

---

## How to Use

1. **Data Cleaning**: Run the Jupyter notebook in the `Data Cleaning` folder to clean the dataset and generate the `train_cleaned.csv` file.
2. **Variable Analysis**: Use the notebook in the `Variable Analysis` folder to visualize the relationships between variables and choose relevant features for modeling.
3. **Traditional Model**: Run the Python code in the `Traditional Model` folder to categorize the scores based on the model got on MATLAB.
4. **Credit Risk Model**: Execute the Jupyter notebook in the `Credit Risk Model` folder to train and evaluate machine learning models on the cleaned data.

---

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `statsmodels` (for traditional models in MATLAB, ensure MATLAB is set up for running scripts)

---

## Authors
- ranix
- silvio
- fredy
- manureymon
  
---

Feel free to contribute to the project or suggest improvements. Happy coding!
