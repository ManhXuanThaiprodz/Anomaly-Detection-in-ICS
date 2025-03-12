import json
import numpy as np
import pdb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .detector import ICSDetector
import pickle

class RFRegressor(ICSDetector):
    """ Random Forest Regressor-based event detection. """
    
    def __init__(self, **kwargs):
        """ Initializes the RFRegressor with default parameters. """
        
        params = {
            'n_estimators': 50,
            'max_depth': None,
            'random_state': 42,
            'history': 50,
            'verbose': 1
        }
        
        for key, item in kwargs.items():
            params[key] = item
        
        self.params = params
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=self.params['n_estimators'],
                                           max_depth=self.params['max_depth'],
                                           random_state=self.params['random_state'],
                                           verbose=self.params['verbose'])
    
    def transform_to_window_data(self, dataset, target, target_size=1):
        """ Transforms data into windows for time series forecasting. """
        data, labels = [], []
        history = self.params['history']

        for i in range(history, len(dataset) - target_size):
            data.append(dataset[i - history:i])
            labels.append(target[i + target_size])
        
        return np.array(data), np.array(labels)
    
    def train(self, Xtrain, Ytrain):
        """ Trains the RF Regressor model with progress display. """
        if self.params['verbose']:
            print("Starting training...")

        # Biến đổi dữ liệu đầu vào

        # Làm phẳng đầu vào
        Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
        Xtrain = self.scaler.fit_transform(Xtrain)
        #Small data
        # X_train_small = Xtrain[0:10]
        # Y_train_small = Ytrain[0:10]
        # self.model.fit(X_train_small, Y_train_small)

        # Huấn luyện mô hình
        self.model.fit(Xtrain, Ytrain)

        if self.params['verbose']:
            print("Training completed.")

    def predict(self, X):
        """ Makes predictions using the trained model. """
        

        # Làm phẳng đầu vào
        X = X.reshape(X.shape[0], -1)
        X = self.scaler.transform(X)

        return self.model.predict(X)
    
    def detect(self, x, theta, window=1):
        """ Performs anomaly detection based on reconstruction errors. """
        
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

        # Đảm bảo predict() trả về cùng kích thước với Ywindow
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
