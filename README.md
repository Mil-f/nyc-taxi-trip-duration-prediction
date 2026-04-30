# NYC Taxi Trip Duration Prediction

## Problem Statement

The goal of this project is to build a machine learning model that predicts the total ride duration of taxi trips in New York City. By learning from features like pickup/dropoff coordinates, time of day, and distance, we can build a strong predictive model.

## Data Source

The dataset used is based on the NYC Taxi Trip Duration competition on Kaggle. It contains variables such as:
- `pickup_datetime`
- `pickup_longitude`, `pickup_latitude`
- `dropoff_longitude`, `dropoff_latitude`
- `passenger_count`
- `vendor_id`

*Note: Data files are excluded from this repository. Ensure you place the `train.csv` and `test.csv` inside `data/split/`.*
The dataset is not included in this repository due to privacy. You can download the original NYC Taxi Trip Duration dataset from Kaggle [NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview)

## Feature Engineering

My approach carefully separates generic feature generation from training-only filtering to prevent data leakage. Key features engineered:
- **Cyclical Time**: `hour_sin`, `hour_cos` to handle the wrap-around of 24-hour clocks.
- **Distances**: `log_distance` (Haversine distance) and `manhattan_km`.
- **Direction**: `bearing_sin`, `bearing_cos` for cyclical representation of the trip bearing.
- **Location Flags**: `pickup_airport` and `dropoff_airport` for trips starting or ending at JFK/LGA.

## Models Used

1. **Ridge Regression**: A linear baseline using `RidgeCV` to handle multi-collinearity and find the optimal alpha.
2. **XGBoost**: A non-linear gradient boosted tree capable of learning complex geographical interactions and time-based splits.

## Results

*The following metrics were evaluated on the validation split:*

| Model      | Validation R² | Validation RMSE |
|------------|---------------|-----------------|
| Ridge      | 0.5918        | 0.5085          |
| XGBoost    | 0.7125        | 0.4268          |

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd nyc-taxi-duration-prediction
   ```

2. **Set up a Virtual Environment (Optional but Recommended)**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Data Placement**:
   Ensure you have downloaded the dataset from Kaggle and placed `train.csv` and `test.csv` inside the `data/split/` directory.

5. **Run the Pipeline**:
   Execute the main script to preprocess the data, train both the Ridge and XGBoost models, evaluate their performance, and save the model artifacts to the `models/` folder.
   ```bash
   python main.py
   ```

6. **Explore the Data**:
   You can also explore the initial data analysis by opening the notebook:
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```
