import numpy as np
import joblib
import xgboost as xgb
from tqdm import tqdm

class XGBoostDetector:
    def __init__(self, **kwargs):
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'history': 100,
            'objective': 'reg:squarederror',  # Sử dụng hồi quy MSE
            'verbose': 1
        }
        params.update(kwargs)
        self.params = params
        self.model = xgb.XGBRegressor(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            learning_rate=self.params['learning_rate'],
            objective=self.params['objective'],
            random_state=self.params['random_state']
        )

    def create_model(self):
        # XGBoost không cần khởi tạo trước khi train
        pass

    def train(self, Xtrain, Ytrain, validation_data=None, **kwargs):
        Xtrain_flat = Xtrain.reshape(Xtrain.shape[0], -1)
        Ytrain_flat = Ytrain.reshape(Ytrain.shape[0], -1)
        print("X_train_flat shape:", Xtrain_flat.shape)
        print("Y_train shape:", Ytrain_flat.shape)
        Xtrain_small = Xtrain_flat[:100]
        Ytrain_small = Ytrain_flat[:100]
        print("XGBoost training...")
        self.model.fit(Xtrain_small, Ytrain_small)
        # event_detector.train(X_train_windowed, Y_train_windowed)
        print("XGBoost training completed!")

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        preds = self.model.predict(X_flat)
        return preds.reshape(X.shape[0], -1)

    def transform_to_window_data(self, dataset, target, target_size=1):
        data = []
        labels = []

        # history = self.params['history']
        history =100

        start_index = history
        end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history, i)
            data.append(dataset[indices])
            labels.append(target[i+target_size])

        return np.array(data), np.array(labels)

    
    def detect(self, x, theta, window = 1, batches=False, eval_batch_size = 4096, **keras_params):
        """ Detection performed based on (smoothed) reconstruction errors.

            x = inputs,
            theta = threshold, attack flagged if reconstruction error > threshold,
            window = length of the smoothing window (default = 1 timestep, i.e. no smoothing),
            average = boolean (default = False), if True the detection is performed
                on the average reconstruction error across all outputs,
            keras_params = parameters for the Keras-based AE prediction.
        """
        reconstruction_error = self.reconstruction_errors(x, batches, eval_batch_size, **keras_params)
        
        # Takes the mean error over all features
        instance_errors = reconstruction_error.mean(axis=1)
        return self.cached_detect(instance_errors, theta, window)

    def cached_detect(self, instance_errors, theta, window = 1):
        """
            Same as detect, but using the errors pre-computed
        """

        # Takes the mean error over all features
        detection = instance_errors > theta

        # If window exceeds one, look for consective detections
        if window > 1:

            detection = np.convolve(detection, np.ones(window), 'same') // window

            # clement: Removing this behavior
            # Backfill the windows (e.g. if idx 255 is 1, all of 255-window:255 should be filled)
            # fill_idxs = np.where(detection)
            # fill_detection = detection.copy()
            # for idx in fill_idxs[0]:
            #     fill_detection[idx - window : idx] = 1
            # return fill_detection

        return detection


    # MSE
    def reconstruction_errors(self, x, batches=False, eval_batch_size = 4096, **keras_params):
        
        if batches:
            
            full_errors = np.zeros((x.shape[0] - self.params['history'] - 1, x.shape[1]))
            idx = 0
            
            while idx < len(x):
                
                Xwindow, Ywindow = self.transform_to_window_data(x[idx: idx + eval_batch_size + self.params['history'] + 1], x[idx:idx + eval_batch_size + self.params['history'] + 1])

                if idx + eval_batch_size > len(full_errors):
                    full_errors[idx:] = (self.predict(Xwindow, **keras_params) - Ywindow)**2                
                else:
                    full_errors[idx:idx+eval_batch_size] = (self.predict(Xwindow, **keras_params) - Ywindow)**2
                idx += eval_batch_size

            return full_errors

        else:
            # LSTM needs windowed data
            Xwindow, Ywindow = self.transform_to_window_data(x, x)
            return (self.predict(Xwindow, **keras_params) - Ywindow)**2

    def save(self, filename):
        joblib.dump(self.model, f"{filename}.pkl")

    def load(self, path):
        self.model = joblib.load(path)
