import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

def predict_eval(model, df, features, name):
    y_pred = model.predict(df[features])
    rmse = np.sqrt(mean_squared_error(df['log_trip_duration'], y_pred))
    r2 = r2_score(df['log_trip_duration'], y_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")

def train_xgb(train, test):
    print("Training XGBoost...")

    categorical_features = [
        'dayofweek', 'month', 'pickup_airport', 'dropoff_airport',
        'is_weekend', 'is_rush_hour', 'is_airport_trip'
    ]
    if 'vendor_id' in train.columns:
        categorical_features.append('vendor_id')
    if 'passenger_count' in train.columns:
        categorical_features.append('passenger_count')

    numeric_features = [
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'log_distance', 'hour_sin', 'hour_cos',
        'manhattan_km', 'bearing_sin', 'bearing_cos',
        'delta_lat', 'delta_lon', 'distance', 'log_distance_x_hour'
    ]

    train_features = categorical_features + numeric_features

    # No scaling for numeric features in XGBoost
    preprocess = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

    xgb_model = Pipeline(steps=[
        ('preprocess', preprocess),
        ('model', XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42
        ))
    ])

    xgb_model.fit(train[train_features], train['log_trip_duration'])
    
    predict_eval(xgb_model, train, train_features, "XGB Train")
    if test is not None and 'log_trip_duration' in test.columns:
        predict_eval(xgb_model, test, train_features, "XGB Validation")

    return xgb_model, train_features
