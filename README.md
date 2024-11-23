# Predicting churns with Spark
This project predicts churn using PySpark on a sample dataset derived from a larger 12GB dataset of the fictitious music service platform, **“Spartify”**
> You can check more from the [Medium blogs](https://medium.com/@hideonhp/leveraging-apache-spark-to-predict-customer-churn-in-music-streaming-services-d9bd89a140dd) for more details

## 1. Motivation
Customer attrition (or erosion) is a critical business issue because the cost of regaining a lost customer is considerably more than investing in keeping the existing one. Forecasting churn means that a company is able to focus on the specific target customer retention strategy where retention programs are executed even as the churn occurs.

For this project, I decided to apply [Apache Spark](https://spark.apache.org/) to analyze the logs of user activity dataset and create a machine learning model to predict users most likely to churn.

## 2. Datasets
- User activity dataset: [Udacity](https://www.udacity.com/)
    > The dataset contains user information (e.g., name, gender, state) and actions (e.g., the song listened to, the type of event, the device used) performed at one or more times.

- This table was created from the Census regions table available on [Cphalpert’s GitHub](https://github.com/cphalpert/census-regions).
    > It shows the relationship between states and geographical area divisions.

Explorations and pilot modeling were conducted using an initial sample of data (~120MB), while the final dataset for machine learning model training consisted of the full 12GB.

## 3. What have done in the notebook
### 3.1. Data Preprocessing
- Load a subset of data from JSON
- Handle missing data

### 3.2. Exploratory Data Analysis (EDA)
- Summary of numerical columns: Basic statistics
- Summary of non-numerical columns: Identify potential categories
- Define churn as the cancellation of service
- Analyze the behavior of churn vs. non-churn users in the following aspects:
    - Usage patterns at different times of the day
    - Usage trends across different days of the week
    - User types (free vs. paid accounts)
    - Interaction types (e.g., adding friends, viewing advertisements, giving thumbs up)
    - Device preferences (e.g., Mac, Windows)
    - Geographic distribution of users (e.g., New England, Pacific regions)
    - Time elapsed from downgrade to churn

### 3.3. Feature Engineering for ML
- Generate user-level features:
    - Most recent user level
    - Time since account creation
    - User's gender
    - User's interaction metrics: number of songs, artists, and sessions
    - Average and standard deviation of songs per artist, per session, and session duration
    - Device type
    - Frequency and proportion of each event type
    - User’s location
- Remove highly correlated features (retain one from each correlated pair)
- Transform features to approximate normal distributions
- Organize feature engineering code for scalability

### 3.4. Machine Learning Pipeline Development
- Split data into training and testing sets
- Define performance evaluation metrics
- Create utility functions for building the cross-validation pipeline, training the model, and assessing model performance
- Evaluate initial models using:
    - Baseline prediction (naive approach)
    - Logistic regression
    - Random forest
    - Gradient-boosted tree

## 4. Results
| **Model**                | **Train Time (s)** | **F1 Train** | **Acc Train** | **F1 Test** | **Acc Test** | **Prediction Time (s)** |
|--------------------------|-------------------|-------------|-------------|------------|------------|-------------------------|
| **Naive Model**          | -                 | -           | -           | **0.6684** | **0.7689** | -                      |
| **Logistic Regression**  | 648.1715          | 0.8685      | 0.8729      | 0.6706     | 0.6818     | 1.3453                 |
| **Random Forest**        | 696.1144          | 0.9165      | 0.9227      | **0.6429** | **0.7500** | 0.9643                 |
| **Gradient-Boosted Tree**| 564.1485          | **1.0**     | **1.0**     | 0.6714     | 0.6591     | **0.7612**             |
> Visualize feature importance for Gradient-Boosted Tree

![Visualize Feature Importance](Visualize_feature_importance.png)

## 5. Final Model
Reason for Selection: Based on the evaluation metrics, the Random Forest model is selected as the final model due to its superior test accuracy (75.0%) and reasonable generalization performance. It balances predictive power with fast prediction times.

## 6. Model Comparison and Final Evaluation after Hypertuning with Grid Search

| Model                     | Accuracy | F1 Score | Training Time (s) | Prediction Time (s) |
|---------------------------|----------|----------|-------------------|---------------------|
| Naive Model               | 0.7689   | 0.6684   | -                 | -                   |
| Logistic Regression       | 0.6818   | 0.6706   | 648.17            | 1.35                |
| Random Forest (Final)     | 0.7588   | 0.6400   | 696.11            | 0.96                |
| Gradient-Boosted Tree     | 0.6591   | 0.6714   | 564.15            | 0.76                |


Key Insights:
- Naive Model: The baseline model provides an accuracy of 0.7689 and an F1 score of 0.6684, setting a reference point.

- Logistic Regression: While it has a relatively fast prediction time (1.35s), its performance (accuracy = 0.6818, F1 score = 0.6706) is not as good as the Random Forest model, which indicates limited generalization capability compared to the Random Forest.

- Random Forest: With the final model (100 trees, max depth = 5), it achieves a good balance of accuracy (0.7588) and F1 score (0.6400), showing good performance while maintaining generalization. It has a moderate training time and a quick prediction time.

- Gradient-Boosted Tree: This model achieved perfect training metrics but overfitted, as seen by the much lower test accuracy (0.6591). Despite fast training and prediction times, this model requires tuning to reduce overfitting.

Conclusion:

The **Random Forest** model, with 100 trees and a max depth of 5, performs the best in terms of test accuracy while striking a balance between training and prediction time. While Gradient-Boosted Trees showed high training performance, they suffered from overfitting, making Random Forest the more reliable option for this dataset.

## 7. Installation

Prototype on Google Colab: The code was developed using Google Colab, with Python version 3. Libraries used include PySpark, Pandas, Seaborn, and Matplotlib.

## 8. Project Files Overview

- Sparkify.ipynb: This notebook contains the exploratory data analysis, data cleaning, and preliminary machine learning model development using a sample of the user activity dataset on a local machine.
- mini_sparkify_event_data.json: A smaller portion of the user activity data, used for initial testing and analysis.
- us census bureau regions and divisions.csv: A CSV file containing data on U.S. census bureau regions and divisions, used for geographical segmentation.

> You can check more from the [Medium blogs](https://medium.com/@hideonhp/leveraging-apache-spark-to-predict-customer-churn-in-music-streaming-services-d9bd89a140dd) for more details
