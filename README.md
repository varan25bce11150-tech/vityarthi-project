# 🏡 California Housing Price Predictor

This project implements a machine learning pipeline to predict the median house value in California districts using the widely-known California Housing dataset. It utilizes a **Random Forest Regressor** and incorporates **feature engineering** and **robust preprocessing** steps.

## 🌟 Features

* **Robust Data Loading:** Automatically loads data from Google Drive or falls back to a public GitHub URL.
* **Data Cleaning:** Handles missing values and removes duplicate entries.
* **Feature Engineering:** Creates derived features to improve model performance (e.g., `rooms_per_household`).
* **Scikit-learn Pipeline:** Uses `ColumnTransformer` and `Pipeline` for seamless data preprocessing (handling numerical and categorical features) and model training.
* **Comprehensive Evaluation:** Provides both **Regression Metrics** (RMSE, R2 Score) and **Classification Metrics** (Accuracy, Precision) based on a median price threshold.
* **Visualization:** Generates a plot comparing actual vs. predicted values.
* **Prediction Function:** Includes a function for making single predictions on new, raw input data.

---

## 🛠 Prerequisites

This project is designed to run in a **Google Colab** environment, which is evident from the `drive.mount` command and the file path structure.

You will need the following Python libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn` (specifically `train_test_split`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `RandomForestRegressor`, and various metrics)

---

## 🚀 Installation and Execution

1.  **Open in Google Colab:** Upload the code to a new Colab notebook.
2.  **Mount Google Drive:** The first cell attempts to mount your Google Drive to access the dataset.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  **Ensure Data Availability:**
    * Place the `housing.csv` file in your Google Drive at the path: `/content/drive/MyDrive/housing.csv`.
    * *(If the file is not found, the script will automatically download it from a public source.)*
4.  **Run All Cells:** Execute all the code cells sequentially.

---

## 💻 Code Overview

### 1. Data Preparation

* The script cleans the data by filling missing `total_bedrooms` with the median value.
* New features are created:
    * `rooms_per_household`
    * `bedrooms_per_room`
    * `population_per_household`

### 2. Preprocessing Pipeline

A `ColumnTransformer` is used to apply appropriate transformations to different column types:
* **Numerical Features:** Passed through (`passthrough`).
* **Categorical Features (`ocean_proximity`):** Transformed using **OneHotEncoder**.

### 3. Model Training

A `Pipeline` combines the preprocessing step and the model:

$$\text{Pipeline} = (\text{Preprocessor}) \rightarrow (\text{RandomForestRegressor})$$

The `RandomForestRegressor` is trained with $\mathbf{n\_estimators=100}$.

---

## ✅ Results and Evaluation

The model achieved strong results on the test data (5% split).

### Regression Performance

| Metric | Value |
| :--- | :--- |
| **Root Mean Squared Error (RMSE)** | **$49,554.49** |
| **R2 Score** | **0.8191** |

### Classification Performance (High/Low Value)

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **87.60%** |
| **Precision** | **0.84** |

### Sample Prediction

Using a sample house in the **NEAR BAY** proximity (with Median Income 8.3252):
