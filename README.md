# Fashion Preference Prediction

This project explores the prediction of fashion styles based on synthetic user preference data. It involves data generation, exploratory data analysis (EDA), data preprocessing, and machine learning model training and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Results and Conclusion](#results-and-conclusion)

## Project Overview

This notebook demonstrates a complete machine learning pipeline for predicting fashion styles. The key steps include:

1.  **Data Generation**: Creating a synthetic dataset of fashion preferences.
2.  **Exploratory Data Analysis (EDA)**: Understanding the distribution and relationships within the data.
3.  **Data Preprocessing**: Encoding categorical features and scaling numerical features.
4.  **Model Training**: Implementing and training various classification models.
5.  **Model Evaluation**: Assessing model performance using accuracy and F1-score.

## Dataset

The synthetic dataset contains 1000 samples with the following features:

-   `age`: Age of the individual (16-25).
-   `favorite_influencer`: Preferred fashion influencer.
-   `preferred_color`: Favorite color.
-   `shopping_frequency`: How often the individual shops.
-   `instagram_hours`: Hours spent on Instagram per day.
-   `tiktok_hours`: Hours spent on TikTok per day.
-   `budget_range`: Budget for fashion items.
-   `preferred_store`: Favorite clothing store.
-   `fashion_style`: The target variable, representing the predicted fashion style (e.g., 'streetwear', 'athleisure', 'vintage', 'minimal', 'y2k').

## Exploratory Data Analysis (EDA)

The EDA section includes visualizations and statistics to understand the dataset:

-   **Dataset Overview**: Basic information about the DataFrame, including data types and non-null counts.
-   **Style Distribution**: Counts of each fashion style in the dataset.
-   **Basic Statistics**: Descriptive statistics for numerical features.
-   **Age Distribution by Fashion Style**: Box plot showing age distribution across different fashion styles.
-   **Budget Distribution by Fashion Style**: Box plot showing budget distribution across different fashion styles.
-   **Color Preferences by Style**: Heatmap illustrating the relationship between preferred colors and fashion styles.
-   **Shopping Frequency by Style**: Box plot showing shopping frequency across different fashion styles.
-   **Social Media Usage by Style**: Bar plot comparing Instagram and TikTok hours across different fashion styles.
-   **Store Preferences Distribution**: Pie chart showing the distribution of preferred stores.

## Data Preprocessing

Before training the models, the data undergoes the following preprocessing steps:

-   **Feature Separation**: The dataset is split into features (X) and the target variable (y).
-   **Label Encoding**: Categorical features (`favorite_influencer`, `preferred_color`, `preferred_store`) are converted into numerical representations using `LabelEncoder`.
-   **Data Splitting**: The data is split into training (80%) and testing (20%) sets, ensuring stratification by the target variable.
-   **Feature Scaling**: Numerical features (`age`, `shopping_frequency`, `instagram_hours`, `tiktok_hours`, `budget_range`) are scaled using `StandardScaler` to normalize their ranges.

## Machine Learning Models

Three different classification models are trained and evaluated:

1.  **Random Forest Classifier**: An ensemble learning method that builds multiple decision trees.
2.  **Logistic Regression**: A linear model for binary and multiclass classification.
3.  **K-Nearest Neighbors (KNN)**: A non-parametric, instance-based learning algorithm.

## Results and Conclusion

The models are evaluated based on their accuracy and F1-score (weighted average). A comparison table and a bar plot visualize the performance of each model.

**Model Performance Comparison:**

| Model               | Accuracy | F1-Score |
| :------------------ | :------- | :------- |
| Random Forest       | 0.885    | 0.883922 |
| Logistic Regression | 0.555    | 0.554122 |
| K-Nearest Neighbors | 0.670    | 0.675235 |

From the results, the **Random Forest Classifier** achieved the highest accuracy and F1-score, indicating its superior performance in predicting fashion styles on this synthetic dataset.

Further improvements could involve more sophisticated feature engineering, hyperparameter tuning, or exploring deep learning models if a larger, more complex dataset were available.

