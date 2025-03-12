import json
import numpy as np
import pdb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from .detector import ICSDetector
import pickle

class XGBoostRegressor(ICSDetector):
    """ XGBoost Regressor-based event detection. """
    
    def __init__(self, **kwargs):
        """ Initializes the XGBoostRegressor with default parameters. """
        
        params = {
            'n_estimators': 100, #100
            'max_depth': 6, #6
            'learning_rate': 0.1,
            'random_state': 42,
            'history': 50,
            'verbose': 2
        }
        
        for key, item in kwargs.items():
            params[key] = item
        
        self.params = params
        self.scaler = StandardScaler()
        self.model = XGBRegressor(n_estimators=self.params['n_estimators'],
                                  max_depth=self.params['max_depth'],
                                  learning_rate=self.params['learning_rate'],
                                  random_state=self.params['random_state'],
                                  verbosity=self.params['verbose'])
    
    def transform_to_window_data(self, dataset, target, target_size=1):
        """ Transforms data into windows for time series forecasting. """
        data, labels = [], []
        history = self.params['history']

        for i in range(history, len(dataset) - target_size):
            data.append(dataset[i - history:i])
            labels.append(target[i + target_size])
        
        return np.array(data), np.array(labels)
    
    def train(self, Xtrain, Ytrain):
        """ Trains the XGBoost Regressor model with progress display. """
        
        if self.params['verbose']:
            print("Starting training XGBoost...")
            
        # Biến đổi dữ liệu đầu vào

        # Làm phẳng đầu vào
        Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
        Xtrain = self.scaler.fit_transform(Xtrain)

        # Small data
        # X_train_small = Xtrain[0:1]
        # Y_train_small = Ytrain[0:1]
        # self.model.fit(X_train_small, Y_train_small)

        # Huấn luyện mô hình
        self.model.fit(Xtrain, Ytrain)

        if self.params['verbose']:
            print("Training XGBoost completed.")
    
    def predict(self, X):
        """ Makes predictions using the trained model. """
        

        # Làm phẳng đầu vào
        X = X.reshape(X.shape[0], -1)
        X = self.scaler.transform(X)

        return self.model.predict(X)
    
    def detect(self, x, theta, window=1):
        """ Performs anomaly detection based on reconstruction errors. """
        
        if len(x) < self.params['history']:
            raise ValueError("Input data too short for detection.")

        reconstruction_error = self.reconstruction_errors(x)
        instance_errors = reconstruction_error.mean(axis=1)

        return self.cached_detect(instance_errors, theta, window)
    
    def cached_detect(self, instance_errors, theta, window=1):
        """ Uses precomputed errors for detection. """
        detection = instance_errors > theta
        if window > 1:
            detection = np.convolve(detection, np.ones(window), 'same') // window
        return detection
    
    def reconstruction_errors(self, x, batches= False):
        
        """ Computes reconstruction errors. """
        Xwindow, Ywindow = self.transform_to_window_data(x, x)

        predictions = self.predict(Xwindow)
        if predictions.shape != Ywindow.shape:
            raise ValueError(f"Prediction shape {predictions.shape} does not match Ywindow shape {Ywindow.shape}")

        return (predictions - Ywindow) ** 2
    def save(self, filename):
        """ Save the trained Random Forest model using pickle. """
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ RF model saved at {filename}.pkl")

    def load(self, filename):
        """ Load the trained Random Forest model using pickle. """
        with open(filename + '.pkl', 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ RF model loaded from {filename}.pkl")

if __name__ == "__main__":
    print("Not a main file.")
