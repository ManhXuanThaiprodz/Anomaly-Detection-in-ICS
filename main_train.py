"""

   Copyright 2020 Lujo Bauer, Clement Fung

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

# Uncomment section if EXACT reproducible results are needed
# rseed = 2021
# from numpy.random import seed
# seed(rseed)
# from tensorflow import set_random_seed
# set_random_seed(rseed)

# Generic python
import argparse
import json
import os
import pdb
import pickle
import sys
import time
from sklearn.multioutput import MultiOutputRegressor
# Ignore ugly futurewarnings from np vs tf.
import warnings
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Data science ML
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve, precision_score, recall_score,classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import metrics
import utils
from data_loader import load_test_data, load_train_data

# Custom packages
from detector import autoencoder, cnn, dnn, gru, identity, linear, lstm, rf,Xgboost
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MLIR_CRASH_REPRODUCER_DIRECTORY'] = "0" 
# check if gpu is available
print(tf.config.list_physical_devices('GPU'))



def train_reconstruction_model(model_type, config, Xtrain, Xval):

    train_params = config["train"]
    model_params = config["model"]

    # define model input size parameter --- needed for AE size
    model_params["nI"] = Xtrain.shape[1]

    if model_type == "AE":
        event_detector = autoencoder.AEED(**model_params)
    elif model_type == "ID":
        event_detector = identity.Identity(**model_params)
    else:
        print(f"Model type {model_type} is not supported.")
        return

    event_detector.create_model()

    event_detector.train(Xtrain, validation_data=(Xval, Xval), **train_params)

    return event_detector


def train_forecast_model(model_type, config, Xtrain, Xval, Ytrain, Yval):

    train_params = config["train"]
    model_params = config["model"]

    # define model input size parameter --- needed for AE size
    model_params["nI"] = Xtrain.shape[2]

    if model_type == "GRU":
        event_detector = gru.GatedRecurrentUnit(**model_params)
    elif model_type == "LSTM":                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == "DNN":
        event_detector = dnn.DeepNN(**model_params)
    elif model_type == "CNN":
        event_detector = cnn.ConvNN(**model_params)
    elif model_type == "LIN":
        event_detector = linear.Linear(**model_params)
    elif model_type == "ID":
        event_detector = identity.Identity(**model_params)
    elif model_type == "RF":
        event_detector = rf.RFRegressor(**model_params)
    else:
        print(f"Model type {model_type} is not supported.")
        return

    event_detector.create_model()

    event_detector.train(Xtrain, Ytrain, validation_data=(Xval, Yval), **train_params)

    return event_detector


def train_forecast_model_by_idxs(model_type, config, Xfull, train_idxs, val_idxs):

    train_params = config["train"]
    model_params = config["model"]

    # define model input size parameter --- needed for AE size
    model_params["nI"] = Xfull.shape[1]

    if model_type == "GRU":
        event_detector = gru.GatedRecurrentUnit(**model_params)
    elif model_type == "LSTM":
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == "DNN":
        event_detector = dnn.DeepNN(**model_params)
    elif model_type == "CNN":
        event_detector = cnn.ConvNN(**model_params)
    elif model_type == "LIN":
        event_detector = linear.Linear(**model_params)
    elif model_type == "ID":
        event_detector = identity.Identity(**model_params)
    elif model_type == "RF":
        event_detector = rf.RFRegressor(**model_params)
    else:
        print(f"Model type {model_type} is not supported.")
        return

    event_detector.create_model()
    
    event_detector.train_by_idx(Xfull, train_idxs, val_idxs,
            validation_data=True,
            **train_params)
    return event_detector

def train_ml_model(model_type, config, Xtrain, Ytrain):
    """
    Hàm tổng quát để huấn luyện các thuật toán Machine Learning khác nhau (RF, XGBoost, v.v.).

    Args:
        model_type (str): Loại mô hình cần huấn luyện (RF, XG).
        config (dict): Cấu hình mô hình.
        Xtrain (np.array): Dữ liệu đầu vào huấn luyện.
        Ytrain (np.array): Nhãn đầu ra tương ứng.

    Returns:
        event_detector: Mô hình ML đã được huấn luyện.
    """
    
    model_params = config["model"]
    
    if model_type == "RF":
        print("Initializing Random Forest Regressor...")
        event_detector = rf.RFRegressor(**model_params)

    elif model_type == "XG":
        print("Initializing XGBoost Regressor...")
        event_detector = Xgboost.XGBoostRegressor(**model_params)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Biến đổi dữ liệu thành dạng cửa sổ (windowed format)
    Xtrain_windowed, Ytrain_windowed = event_detector.transform_to_window_data(Xtrain, Ytrain)

    print("X_train_windowed shape:", Xtrain_windowed.shape)
    print("Y_train_windowed shape:", Ytrain_windowed.shape)

    # Huấn luyện mô hình
    event_detector.train(Xtrain_windowed, Ytrain_windowed)

    print(f"✅ {model_type} Training Completed!")
    
    return event_detector


def hyperparameter_search(
    event_detector,
    model_type,
    config,
    Xval,
    Xtest,
    Ytest,
    dataset_name,
    val_idxs=None,
    test_split=0.7,
    run_name="results",
    verbose=1,
):
    print("Xval shape: ", Xval.shape)
    model_name = config["name"]
    do_batches = False

    Ytest = Ytest.astype(int)
    Xtest_val, Xtest_test, Ytest_val, Ytest_test = utils.custom_train_test_split(
        dataset_name, Xtest, Ytest, test_size=test_split, shuffle=False
    )

    if not model_type == "AE":

        history = event_detector.params["history"]

        # Clip the prediction to match LSTM prediction window
        Ytest_test = Ytest_test[history + 1 :]
        Ytest_val = Ytest_val[history + 1 :]
        do_batches = True

    ##### Cross Validation
    if val_idxs is None:
        validation_errors = event_detector.reconstruction_errors(
            Xval, batches=do_batches
        )
    else:
        validation_errors = utils.reconstruction_errors_by_idxs(
            event_detector, Xval, val_idxs, history
        )

    #MSE
    test_errors = event_detector.reconstruction_errors(Xtest_val, batches=do_batches)
    test_instance_errors = test_errors.mean(axis=1)
    print("MSE: ", test_instance_errors)

    # Default to empty dict. Will still do F1 for window=1, ptile=95%
    grid_config = config.get("grid_search", dict())

    cutoffs = grid_config.get("percentile", [0.95])
    windows = grid_config.get("window", [1])
    eval_metrics = grid_config.get("metrics", ["F1"])

    firstPlotsError = True
    firstNpysError = True

    for metric in eval_metrics:

        # FPR is a negative metric (lower is better)
        negative_metric = metric == "false_positive_rate"

        # FPR is a negative metric (lower is better)
        if negative_metric:
            best_metric = 1
        else:
            best_metric = -1000

        best_percentile = 0
        best_window = 0
        metric_vals = np.zeros((len(cutoffs), len(windows)))
        metric_func = metrics.get(metric)

        for percentile_idx in range(len(cutoffs)):

            percentile = cutoffs[percentile_idx]

            # set threshold as quantile of average reconstruction error
            theta = np.quantile(validation_errors.mean(axis=1), percentile)

            for window_idx in range(len(windows)):

                window = windows[window_idx]

                # Yhat = event_detector.detect(Xtest, theta = theta, window = window, batches=True)
                Yhat = event_detector.cached_detect(
                    test_instance_errors, theta=theta, window=window
                )
                # Yhat = Yhat[window-1:].astype(int)
                print("Yhat", Yhat)
                print("Ytest_val", Ytest_val)
                Yhat_trunc, Ytest_trunc = utils.normalize_array_length(Yhat, Ytest)
                choice_value = metric_func(Yhat, Ytest_val)
                print(choice_value)

                if verbose > 0:
                    print(
                        "{} is {:.3f} at theta={:.3f}, percentile={:.4f}, window={}".format(
                            metric, choice_value, theta, percentile, window
                        )
                    )

                # FPR is a negative metric (lower is better)
                if negative_metric:
                    if choice_value < best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window
                else:
                    if choice_value > best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window

                if grid_config.get("save-metric-info", False):
                    metric_vals[percentile_idx, window_idx] = choice_value

                if grid_config.get("detection-plots", False):

                    fig_detect, ax_detect = plt.subplots(figsize=(20, 4))

                    ax_detect.plot(Yhat, color="0.1", label="predicted state")
                    ax_detect.plot(
                        Ytest_val, color="r", alpha=0.75, lw=2, label="real state"
                    )
                    ax_detect.fill_between(
                        np.arange(len(Yhat)), 0, Yhat.astype(int), color="0.1"
                    )
                    ax_detect.set_title(
                        "Detection trajectory on test dataset, {}, percentile={:.3f}, window={}".format(
                            model_type, percentile, window
                        ),
                        fontsize=14,
                    )
                    ax_detect.set_yticks([0, 1])
                    ax_detect.set_yticklabels(["NO ATTACK", "ATTACK"])
                    ax_detect.legend(fontsize=12, loc=2)
                    try:
                        fig_detect.savefig(
                            f"plots/{run_name}/{model_name}-{percentile}-{window}.pdf"
                        )
                        if firstPlotsError:
                            print(
                                f"Saving plots for model {model_name} to plots/{run_name}"
                            )
                            firstPlotsError = False
                    except FileNotFoundError:
                        fig_detect.savefig(
                            f"plots/results/{model_name}-{percentile}-{window}.pdf"
                        )
                        if firstPlotsError:
                            print(
                                f"Directory plots/{run_name}/ not found, saving plots for model {model_name} to plots/results/ instead"
                            )
                            firstPlotsError = False
                    plt.close(fig_detect)

            if grid_config.get("save-theta", False):
                try:
                    pickle.dump(
                        theta,
                        open(
                            f"models/{run_name}/{model_name}-{percentile}-theta.pkl",
                            "wb",
                        ),
                    )
                    print(
                        f"Saved theta to models/{run_name}/{model_name}-{percentile}-theta.pkl"
                    )
                except FileNotFoundError:
                    pickle.dump(
                        theta,
                        open(
                            f"models/results/{model_name}-{percentile}-theta.pkl", "wb"
                        ),
                    )
                    print(
                        f"Directory models/{run_name}/ not found, saved theta to models/results/{model_name}-{percentile}-theta.pkl instead"
                    )

        print(
            "Best metric ({}) is {:.3f} at percentile={:.5f}, window {}".format(
                metric, best_metric, best_percentile, best_window
            )
        )

        # Final test performance
        final_test_errors = event_detector.reconstruction_errors(
            Xtest_test, batches=do_batches
        )
        final_test_instance_errors = final_test_errors.mean(axis=1)

        best_theta = np.quantile(validation_errors.mean(axis=1), best_percentile)
        event_detector.save_detection_params(
            best_theta=best_theta, best_window=best_window
        )

        final_Yhat = event_detector.best_cached_detect(final_test_instance_errors)
        # final_Yhat = final_Yhat[best_window-1:].astype(int)

        # confusion matrix 
        cm = confusion_matrix(Ytest_test, final_Yhat)
        # Chuyển thành DataFrame và lưu vào CSV
        df_cm = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        df_cm.to_csv(f"{model_name}-confusion_matrix.csv", index=True)

        print("Confusion matrix đã được lưu vào 'confusion_matrix.csv'")


        metric_func = metrics.get(metric)
        final_value = metric_func(final_Yhat, Ytest_test)
        print(
            "Final {} is {:.3f} at percentile={:.5f}, window {}".format(
                metric, final_value, best_percentile, best_window
            )
        )
        f1_micro = f1_score(Ytest_test, final_Yhat, average="micro")
        # output f1 score
        report = classification_report(Ytest_test, final_Yhat, digits=4)
        # roc curve
        fpr, tpr, _ = roc_curve(Ytest_test, final_Yhat)
        # precision-recall curve
        precision, recall, _ = precision_recall_curve(Ytest_test, final_Yhat)

        # plot roc curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        ax_roc.plot(fpr, tpr, color="b", label="ROC curve")

        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()

        try:
            fig_roc.savefig(f"plots/{run_name}/{model_name}-roc.pdf")
        except FileNotFoundError:
            fig_roc.savefig(f"plots/results/{model_name}-roc.pdf")
            if firstPlotsError:
                print(
                    f"Directory plots/{run_name}/ not found, saving ROC curve for model {model_name} to plots/results/ instead"
                )
                firstPlotsError = False
        
        plt.close(fig_roc)


        # define ML model
        

        
        
        if grid_config.get("save-metric-info", False):
            try:
                np.save(f"npys/{run_name}/{model_name}-{metric}.npy", metric_vals)
                print(f"Saved metric at npys/{run_name}/{model_name}-{metric}.npy")
            except FileNotFoundError:
                np.save(f"npys/results/{model_name}-{metric}.npy", metric_vals)
                if firstPlotsError:
                    print(
                        f"Directory npys/{run_name}/ not found, saved metric {model_name}-{metric}.npy to npys/results/ instead"
                    )
                    firstPlotsError = False

    return event_detector


def save_model(event_detector, config, run_name="results"):
    model_name = config["name"]
    directory = f"models/{run_name}"
    filename = f"{directory}/{model_name}"

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(directory, exist_ok=True)

    try:
        # Nếu là mô hình ML (Random Forest hoặc XGBoost)
        if isinstance(event_detector, (rf.RFRegressor, Xgboost.XGBoostRegressor)):
            with open(filename + ".pkl", "wb") as f:
                pickle.dump(event_detector.model, f)
            print(f"✅ Model saved at {filename}.pkl")

        # Nếu là mô hình Deep Learning (Keras)
        else:
            event_detector.inner.save(filename + ".h5")
            print(f"✅ Model saved at {filename}.h5")

    except Exception as e:
        print(f"⚠️ Error saving model: {e}")



# functions
def load_saved_model(model_type, run_name, model_name):
    """Load stored model."""

    if model_type == "ID":
        return identity.Identity()

    # load params and create event detector
    try:
        with open(f"models/{run_name}/{model_name}.json") as fd:
            model_params = json.load(fd)
        model_filename = f"models/{run_name}/{model_name}.h5"
    except FileNotFoundError:
        print(
            f"Unable to find models/{run_name}/{model_name}.json, checking models/results/..."
        )
        try:
            with open(f"models/results/{model_name}.json") as fd:
                model_params = json.load(fd)
            print(
                f"Using {model_name}.json and {model_name}.h5 found in models/results/"
            )
            print(
                "Note: we recommend separate directories to avoid writing over experiments"
            )
            model_filename = f"models/results/{model_name}.h5"
        except FileNotFoundError:
            raise SystemExit(
                f"Unable to find model {model_name}. Ensure you have trained the model first"
            )

    if model_type == "AE":
        event_detector = autoencoder.AEED(**model_params)
    elif model_type == "GRU":
        event_detector = gru.GatedRecurrentUnit(**model_params)
    elif model_type == "CNN":
        event_detector = cnn.ConvNN(**model_params)
    elif model_type == "DNN":
        event_detector = dnn.DeepNN(**model_params)
    elif model_type == "LSTM":
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == "LIN":
        event_detector = linear.Linear(**model_params)
    else:
        raise SystemExit(f"Model type {model_type} is not supported.")

    # load keras model
    event_detector.inner = tf.keras.models.load_model(model_filename)

    return event_detector


def parse_arguments():

    parser = utils.get_argparser()

    parser.add_argument(
    "--rf_n_estimators",
    default=100,
    type=int,
    help="Number of trees in Random Forest (n_estimators)"
    )   


    parser.add_argument(
    "--rf_model_params_n_estimators",
    default=100,
    type=int,
    help="Number of trees (n_estimators) for Random Forest model"
    )

    parser.add_argument(
        "--rf_model_params_history", 
        default=100, 
        type=int, 
        help="History window size for Random Forest"
    )
    ### Train Params
    parser.add_argument(
        "--train_params_epochs", default=100, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--train_params_batch_size",
        default=512,
        type=int,
        help="Training batch size. Note: MUST be larger than history/window values given",
    )
    parser.add_argument(
        "--train_params_no_callbacks",
        action="store_true",
        help="Remove callbacks like early stopping",
    )

    # Hyperparameters
    parser.add_argument(
        "--detect_params_percentile",
        default=[
            0.95,
            0.96,
            0.97,
            0.98,
            0.99,
            0.991,
            0.992,
            0.993,
            0.994,
            0.995,
            0.996,
            0.997,
            0.998,
            0.999,
            0.9995,
            0.99995,
        ],
        nargs="+",
        type=float,
        help="Percentiles to look over",
    )
    parser.add_argument(
        "--detect_params_windows",
        default=[1, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        nargs="+",
        type=int,
        help="Windows to look over",
    )
    parser.add_argument(
        "--detect_params_metrics",
        default=["F1"],
        nargs="+",
        type=str,
        help="Metrics to look over",
    )
    parser.add_argument(
        "--detect_params_test_split",
        default=0.7,
        type=float,
        help="Split for testing/validation of detection hyperparameters. Default is 0.7 (hyperparameters evaluated on 30%% of test data, final testing on 70%%.) ",
    )

    # saving items
    parser.add_argument(
        "--detect_params_plots",
        action="store_true",
        help="Make detection plots for each hyperparameter setting",
    )
    parser.add_argument(
        "--detect_params_save_npy",
        action="store_true",
        help="Save the metric values in an npy",
    )
    parser.add_argument(
        "--detect_params_save_theta",
        action="store_true",
        help="Save theta thresholds in a pkl",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    model_type = args.model
    dataset_name = args.dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    # Define training parameters
    ae_train_params = {
        "verbose": 1,
        "batch_size": args.train_params_batch_size,
        "epochs": args.train_params_epochs,
        "use_callbacks": not args.train_params_no_callbacks,
    }

    large_train_params = {
        "batch_size": args.train_params_batch_size,
        "epochs": args.train_params_epochs,
        "use_callbacks": not args.train_params_no_callbacks,
        "steps_per_epoch": 0,
        "validation_steps": 0,
        "verbose": 1,
    }

    config = {
        "grid_search": {
            "percentile": args.detect_params_percentile,
            "window": args.detect_params_windows,
            "metrics": args.detect_params_metrics,
            "pr-plot": False,
            "detection-plots": args.detect_params_plots,
            "save-metric-info": args.detect_params_save_npy,
            "save-theta": args.detect_params_save_theta,
        }
    }

    run_name = args.run_name
    test_split = args.detect_params_test_split
    utils.update_config_model(args, config, model_type, dataset_name)
    print("NAme  ", config["name"])
    model_name = config["name"]
    print("Model name", model_name)

    Xfull, sensor_cols = load_train_data(dataset_name)
    Xtest, Ytest, _ = load_test_data(dataset_name)
    print("Xfull shape: ", Xfull.shape)
    print("Xtest shape: ", Xtest.shape)
    print("Ytest shape: ", Ytest.shape)

    # X_train_windowed, Y_train_windowed = utils.transform_to_window_data(Xfull, Xfull, history=100)
    #X_test_windowed, Y_test_windowed = utils.transform_to_window_data(Xtest, Ytest, history=100)

    # print("X_train_windowed shape: ", X_train_windowed.shape)
    # print("Y_train_windowed shape: ", Y_train_windowed.shape)
    # print("X_test_windowed shape: ", X_test_windowed.shape)
    # print("Y_test_windowed shape: ", Y_test_windowed.shape)

    shuffle = True
    by_idx = True

    model_params = config["model"]


    

    

    # Updates training parameters such as batch size, learning rate, etc.
    if model_type == "AE":
        config.update({"train": ae_train_params})
        Xtrain, Xval, _, _ = train_test_split(
            Xfull, Xfull, test_size=0.2, random_state=42, shuffle=True
        )
        event_detector = train_reconstruction_model(model_type, config, Xtrain, Xval)

        # Search for the best tuning of the window and theta parameters
        hyperparameter_search(
            event_detector,
            model_type,
            config,
            Xval,
            Xtest,
            Ytest,
            dataset_name,
            test_split=test_split,
            run_name=run_name,
            verbose=0,
        )
    #elif model_type =="XG":
        ################
    
    
        # # Xtrain_flat = X_train_windowed.reshape(X_train_windowed.shape[0], -1)
        # # Ytrain_flat = Y_train_windowed.reshape(Y_train_windowed.shape[0], -1)
        # # Xtrain_small = Xtrain_flat[:100]
        # # Ytrain_small = Ytrain_flat[:100]
        # # print("XGBoost training...")
        # # event_detector.fit(Xtrain_small, Ytrain_small)
        # # print("XGBoost training completed!")
        # event_detector = XGBoostDetector(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity =3)
        # event_detector.train(X_train_windowed, Y_train_windowed)
        # # Ytest = Ytest.astype(int)
        # # Xtest_val, Xtest_test, Ytest_val, Ytest_test = utils.custom_train_test_split(
        # #     dataset_name, Xtest, Ytest, test_size=test_split, shuffle=False
        # # )
        # #     #MSE
        # # test_errors = event_detector.reconstruction_errors(Xtest_val, batches=True)
        # # test_instance_errors = test_errors.mean(axis=1)
        # # print("MSE: ", test_instance_errors)

        # hyperparameter_search(
        #         event_detector,
        #         model_type,
        #         config,
        #         Xfull,
        #         Xtest,
        #         Ytest,
        #         dataset_name,
        #         test_split=test_split,
        #         run_name=run_name,
        #         verbose=0,
        #     )
        ###############

    elif model_type in ["RF", "XG"]:  # Nếu là thuật toán ML (RF hoặc XGBoost)
        print(f"Training {model_type} Model...")

        event_detector = train_ml_model(model_type, config, Xfull, Xfull)

        # Thực hiện tìm kiếm tham số tối ưu
        hyperparameter_search(
            event_detector,
            model_type,
            config,
            Xfull,
            Xtest,
            Ytest,
            dataset_name,
            test_split=test_split,
            run_name=run_name,
            verbose=1,
        )

    else:

        history = config["model"]["history"]

        train_idxs, val_idxs = utils.train_val_history_idx_split(Xfull, history)

        large_train_params["steps_per_epoch"] = (
            len(train_idxs) // large_train_params["batch_size"]
        )
        large_train_params["validation_steps"] = (
            len(val_idxs) // large_train_params["batch_size"]
        )
        config.update({"train": large_train_params})

        event_detector = train_forecast_model_by_idxs(
            model_type, config, Xfull, train_idxs, val_idxs
        )
        print("-----------------------------------------------")
        print(val_idxs)
        # Search for the best tuning of the window and theta parameters
        hyperparameter_search(
            event_detector,
            model_type,
            config,
            Xfull,
            Xtest,
            Ytest,
            dataset_name,
            val_idxs=val_idxs,
            test_split=test_split,
            run_name=run_name,
            verbose=0,
        )

    save_model(event_detector, config, run_name=run_name)




    print("Finished!")
