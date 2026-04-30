import os
import pandas as pd
import joblib
from src.features import add_features, clean_train
from src.train_ridge import train_ridge
from src.train_xgb import train_xgb

def main():
    print("Loading data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'data', 'split', 'train.csv')
    test_path = os.path.join(base_dir, 'data', 'split', 'test.csv')
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please ensure data is in the correct directory.")
        return

    train = pd.read_csv(train_path)
    
    test = None
    if os.path.exists(test_path):
        test = pd.read_csv(test_path)
    else:
        print(f"Warning: {test_path} not found. Will proceed without test evaluation.")

    print("\n--- Feature Engineering ---")
    train = add_features(train)
    train = clean_train(train)
    
    if test is not None:
        test = add_features(test)

    print("\n--- Model Training: Ridge Baseline ---")
    ridge_model, ridge_features = train_ridge(train, test)
    
    print("\n--- Model Training: XGBoost ---")
    xgb_model, xgb_features = train_xgb(train, test)

    print("\n--- Saving Models ---")
    os.makedirs('models', exist_ok=True)
    joblib.dump(ridge_model, os.path.join('models', 'ridge_model.joblib'))
    joblib.dump(xgb_model, os.path.join('models', 'xgb_model.joblib'))
    print("Models saved in the models/ directory.")

if __name__ == '__main__':
    main()
