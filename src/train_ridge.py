import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV

def predict_eval(model, df, features, name):
    y_pred = model.predict(df[features])
    rmse = np.sqrt(mean_squared_error(df['log_trip_duration'], y_pred))
    r2 = r2_score(df['log_trip_duration'], y_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")

def train_ridge(train, test):
    print("Training Ridge Baseline...")
    
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

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
    ], remainder='passthrough')

    alphas = np.logspace(-3, 3, 13)

    ridge_pipe = Pipeline(steps=[
        ('preprocess', column_transformer),
        ('model', RidgeCV(alphas=alphas))
    ])

    ridge_pipe.fit(train[train_features], train['log_trip_duration'])
    
    print(f"Best Alpha found: {ridge_pipe.named_steps['model'].alpha_}")
    
    predict_eval(ridge_pipe, train, train_features, "Ridge Train")
    if test is not None and 'log_trip_duration' in test.columns:
        predict_eval(ridge_pipe, test, train_features, "Ridge Validation")

    return ridge_pipe, train_features
