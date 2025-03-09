import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

class RandomForestDetector:
    def __init__(self, **kwargs):
        params = {
            'n_estimators': 100,
            'random_state': 42,
            'history': 100,
            'verbose': 1
        }
        params.update(kwargs)
        self.params = params
        self.model = RandomForestRegressor(
            n_estimators=self.params['n_estimators'],
            random_state=self.params['random_state']
        )

    def create_model(self):
        # Random Forest không cần khởi tạo model trước khi train
        pass

    def train(self, Xtrain, Ytrain, validation_data=None, **kwargs):
        Xtrain_flat = Xtrain.reshape(Xtrain.shape[0], -1)
        Ytrain_flat = Ytrain.reshape(Ytrain.shape[0], -1)
        print("X_train_flat shape",Xtrain_flat.shape)
        print("Y_train shape:", Ytrain_flat.shape)
        print("Training Random Forest...")
        self.model.fit(Xtrain_flat, Ytrain_flat)
        
        print("Training Completed!")

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        preds = self.model.predict(X_flat)
        return preds.reshape(X.shape[0], -1)

    def reconstruction_errors(self, X, batches=False):
        predictions = self.predict(X)
        Y_true = X[:, -1, :]  # target là timestep tiếp theo giống như DL
        errors = (predictions - Y_true) ** 2
        
        return errors

    def cached_detect(self, errors, theta, window=1):
        # Hàm này giống các DL khác, biến đổi lỗi thành labels bất thường
        return (errors > theta).astype(int)

    def save(self, filename):
        joblib.dump(self.model, f"{filename}.pkl")

    def load(self, path):
        self.model = joblib.load(path)
