import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import joblib
from sklearn.pipeline import Pipeline

# Load test features dynamically
test_features_path = "outputs/datasets/processed/final/x_test_final.csv"
try:
    test_features = pd.read_csv(test_features_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Test features file not found at {test_features_path}.")

# Load the scaler used during training
scaler_path = "outputs/models/scaler2.pkl"
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

def preprocess_unseen_data(X):
    """
    Preprocess data assuming all required columns are included.

    Parameters:
        X: DataFrame containing all required columns.

    Returns:
        Processed DataFrame ready for prediction
    """
    print("Input to preprocess_unseen_data:")
    print(X.head())

    # Create a copy of the input data
    processed_data = X.copy()

    # Derive new features
    current_year = datetime.datetime.now().year
    processed_data["num__Age"] = current_year - processed_data["YearBuilt"]
    processed_data["num__LivingLotRatio"] = processed_data["LotArea"] / processed_data["GrLivArea"].replace(0, 1)
    processed_data["num__FinishedBsmtRatio"] = processed_data["BsmtFinSF1"] / processed_data["TotalBsmtSF"].replace(0, 1)
    processed_data["num__OverallScore"] = processed_data["OverallQual"] + processed_data["OverallCond"]
    processed_data["cat__HasPorch"] = (processed_data["OpenPorchSF"].astype(float) > 0).astype(float)

    # Drop extra features
    extra_features = {'TotalBsmtSF'}
    processed_data.drop(columns=extra_features.intersection(processed_data.columns), inplace=True)

    # Rename columns to match the test features
    processed_data.rename(
        columns={
            "1stFlrSF": "num__1stFlrSF",
            "2ndFlrSF": "num__2ndFlrSF",
            "BedroomAbvGr": "num__BedroomAbvGr",
            "BsmtFinSF1": "num__BsmtFinSF1",
            "BsmtUnfSF": "num__BsmtUnfSF",
            "GarageArea": "num__GarageArea",
            "GarageYrBlt": "num__GarageYrBlt",
            "GrLivArea": "num__GrLivArea",
            "LotArea": "num__LotArea",
            "LotFrontage": "num__LotFrontage",
            "MasVnrArea": "num__MasVnrArea",
            "OpenPorchSF": "num__OpenPorchSF",
            "OverallCond": "num__OverallCond",
            "OverallQual": "num__OverallQual",
            "YearBuilt": "num__YearBuilt",
            "YearRemodAdd": "num__YearRemodAdd",
        },
        inplace=True,
    )

    # Debugging Outputs
    print("Processed Data Columns:")
    print(processed_data.columns)

    print("\nTest Features Columns:")
    print(test_features.columns)

    # Reorder columns to match training data
    processed_data = processed_data[test_features.columns]
    assert list(processed_data.columns) == list(test_features.columns), "Column alignment mismatch!"
    print("Processed data before scaling:")
    print(processed_data.head())
    print(processed_data.columns)

    # Apply scaling and encoding using the laoded scaler
    numeric_features = [col for col in processed_data.columns if col.startswith("num__")]
    processed_data[numeric_features] = scaler.transform(processed_data[numeric_features])

    print("Processed data after scaling and encoding:")
    print(processed_data.head())

    return processed_data

# Define a wrapper for the preprocess_unseen_data function
def preprocess_general_wrapper(X):
    return preprocess_unseen_data(X)

# Create a preprocessor pipeline using the loaded scaler
preprocessor = Pipeline(steps=[
    ("preprocessing_function", FunctionTransformer(preprocess_general_wrapper))
])

# Fit the preprocessor on test-features
preprocessor.fit(test_features)