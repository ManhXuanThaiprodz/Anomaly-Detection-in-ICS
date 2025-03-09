import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_test_data, load_train_data

def create_rf_dataset_for_test(X_full, history=50):
    """
    Create training dataset for Random Forest model using sliding window approach for both training and test data.
    
    Parameters:
    X_full: Input data (n_samples, n_features)
    history: Number of time steps to use as input window
    
    Returns:
    X_train: Input features with sliding window
    Y_train: Target values for prediction
    """
    all_idxs = np.arange(history, len(X_full)-1)
    X_data, Y_data = [], []

    for i in range(len(X_full) - history):
        window = X_full[i:i+history].flatten()  # Flattening the window
        X_data.append(window)
        Y_data.append(X_full[i + history])

    return np.array(X_data), np.array(Y_data)

# Adjust `X_test` and `X_train` with the same windowing





def create_rf_dataset(X_full, history=50):
    """
    Create training dataset for Random Forest model using sliding window approach.
    
    Parameters:
    X_full: Input data (n_samples, n_features)
    history: Number of time steps to use as input window
    
    Returns:
    X_train: Input features with sliding window
    Y_train: Target values for prediction
    """
    all_idxs = np.arange(history, len(X_full)-1)
    X_train, Y_train = [], []

    # Create sliding window dataset
    for i in range(len(X_full) - history):
        window = X_full[i:i+history].flatten()  # Flattening the window
        X_train.append(window)
        
        # Predict the next time step
        Y_train.append(X_full[i + history])

    return np.array(X_train), np.array(Y_train)

def train_rf_model(X_train, Y_train, X_test, Y_test, n_estimators=1):
    """
    Train and evaluate a Random Forest model.
    
    Parameters:
    X_train: Features for training
    Y_train: Targets for training
    X_test: Features for testing
    Y_test: Targets for testing
    n_estimators: Number of trees in Random Forest
    
    Returns:
    model: Trained Random Forest model
    """
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, Y_train)

    # Predict on test data
    Y_train_pred = rf_model.predict(X_train)
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    print(f'Training Mean Squared Error (MSE): {train_mse}')

    Y_test_pred = rf_model.predict(X_test)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    print(f'Test Mean Squared Error (MSE): {test_mse}')
    
    return rf_model

def main():
    # Example data loading (replace with your actual data loading process)
    dataset_name = "BATADAL"
    X_full, _ = load_train_data(dataset_name)
    X_test, Y_test, _ = load_test_data(dataset_name)

    print("Shape of X_full:", X_full)
    print(Y_test)

    
    # print("Shape of X_test:", X_test.shape)
    # print("Shape of Y_test:", Y_test.shape)
    # print("Shape of X_full:", X_full.shape)



    # # Create RF dataset with sliding window approach
    # history = 50
    # X_train, Y_train = create_rf_dataset(X_full, history)
    # print("Shape of X_train:", X_train.shape)
    # print("Shape of Y_train:", Y_train.shape)

    # history = 50

    # X_test, Y_test = create_rf_dataset_for_test(X_test, history)
    # # Scale the data
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
    # X_test_scaled = scaler.transform(X_test[:len(X_train)])  # Transform test data

    # print("Shape of X_train_scaled:", X_train_scaled.shape)
    # print("Shape of X_test_scaled:", X_test_scaled.shape)

    # # Train and evaluate Random Forest model
    # rf_model = train_rf_model(X_train_scaled, Y_train, X_test_scaled, Y_test)

if __name__ == "__main__":
    main()
