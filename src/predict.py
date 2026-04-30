import pandas as pd
import numpy as np
import os
import joblib

def predict(model_path, test_data_path, output_path, features):
    print(f"Loading test data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    print("Generating predictions...")
    log_predictions = model.predict(test_df[features])
    predictions = np.expm1(log_predictions)  # Inverse of log1p
    
    submission = pd.DataFrame({
        'id': test_df.get('id', test_df.index),
        'trip_duration': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # Example usage:
    pass
