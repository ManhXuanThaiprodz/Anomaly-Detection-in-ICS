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
import argparse
import json
import pickle
import os
import pdb
import sys

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# Data and ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Custom local packages
from data_loader import load_train_data, load_test_data
from main_train import load_saved_model
import metrics
import utils

def eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics=['F1'], percentile=0.995, window=1, plot=False, best=False):

    if best:
        Yhat = event_detector.best_cached_detect(test_errors)
        used_window = event_detector.params['best_window']
        used_theta = event_detector.params['best_theta']
        Yhat = Yhat[used_window-1:].astype(int)
    else:
        theta = np.quantile(val_errors, percentile)
        Yhat = event_detector.cached_detect(test_errors, theta = theta, window = window)
        Yhat = Yhat[window-1:].astype(int)
        used_window = window
        used_theta = theta

    Yhat_trunc, Ytest_trunc = utils.normalize_array_length(Yhat, Ytest)

    # Final test performance
    for metric in eval_metrics:

        metric_func = metrics.get(metric)
        final_value = metric_func(Yhat_trunc, Ytest_trunc)
        if best:
            print(f'Best: At theta={used_theta}, window={used_window}, {metric}={final_value}')
        else:
            print(f'At theta={used_theta}, window={used_window}, {metric}={final_value}')

    # for debugging purposes
    if plot:
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(-1 * Yhat_trunc, color = '0.25', label = 'Predicted')
        ax.plot(Ytest_trunc, color = 'lightcoral', alpha = 0.75, lw = 2, label = 'True Label')
        ax.fill_between(np.arange(len(Yhat_trunc)), -1 * Yhat_trunc, 0, color = '0.25')
        ax.fill_between(np.arange(len(Ytest_trunc)), 0, Ytest_trunc, color = 'lightcoral')
        ax.set_yticks([-1,0,1])
        ax.set_yticklabels(['Predicted','Benign','Attacked'])
        ax.set_title(f'Detection trajectory theta={used_theta}, metric={final_value}, percentile={percentile}, window={used_window}', fontsize = 36)
        fig.savefig(f'eval-detection.pdf')

    return Yhat_trunc, Ytest_trunc

# Compare two settings side by side
def eval_demo(event_detector, model_type, config, val_errors, test_errors, Ytest, eval_metrics=['F1'], run_name='results', include_best = True):

    model_name = config['name']
    eval_config = config['eval']

    if include_best:
        eval_config.append('best')

    # Can't use indexed subplots when length is 1
    if len(eval_config) < 2:

        fig, ax = plt.subplots(figsize=(20, 4))

        if eval_config[0] == 'best':
            Yhat, Ytest = eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics, best=True)
            ax.set_title(f'Detection trajectory on test dataset with best parameters', fontsize = 36)
        else:
            percentile = eval_config[0]['percentile']
            window = eval_config[0]['window']
            ax.set_title(f'Detection trajectory on test dataset, percentile={percentile}, window={window}', fontsize = 36)
            Yhat, Ytest = eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics, percentile=percentile, window=window)

        ax.plot(-1 * Yhat, color = '0.25', label = 'Predicted')
        ax.plot(Ytest, color = 'lightcoral', alpha = 0.75, lw = 2, label = 'True Label')
        ax.fill_between(np.arange(len(Yhat)), -1 * Yhat, 0, color = '0.25')
        ax.fill_between(np.arange(len(Ytest)), 0, Ytest, color = 'lightcoral')
        ax.set_yticks([-1,0,1])
        ax.set_yticklabels(['Predicted','Benign','Attacked'])

    else:

        fig, ax = plt.subplots(len(eval_config), figsize=(20, 4 * len(eval_config)))

        for i in range(len(eval_config)):

            if eval_config[i] == 'best':
                Yhat, Ytest = eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics, best=True)
                ax[i].set_title(f'Detection trajectory on test dataset, with best parameters', fontsize = 36)
            else:
                percentile = eval_config[i]['percentile']
                window = eval_config[i]['window']
                ax[i].set_title(f'Detection trajectory on test dataset, percentile={percentile}, window={window}', fontsize = 36)
                Yhat, Ytest = eval_test(event_detector, model_type, val_errors, test_errors, Ytest, eval_metrics, percentile=percentile, window=window)

            ax[i].plot(-1 * Yhat, color = '0.25', label = 'Predicted')
            ax[i].plot(Ytest, color = 'lightcoral', alpha = 0.75, lw = 2, label = 'True Label')
            ax[i].fill_between(np.arange(len(Yhat)), -1 * Yhat, 0, color = '0.25')
            ax[i].fill_between(np.arange(len(Ytest)), 0, Ytest, color = 'lightcoral')
            ax[i].set_yticks([-1,0,1])
            ax[i].set_yticklabels(['Predicted','Benign','Attacked'])

    plt.tight_layout(rect=[0, 0, 1, 0.925])
    try:
        plt.savefig(f'plots/{run_name}/{model_name}-compare.pdf')
        print(f"Saved plot {model_name}-compare.pdf to plots/{run_name}/")
    except FileNotFoundError:
        plt.savefig(f'plots/results/{model_name}-compare.pdf')
        print(f"Unable to find plots/{run_name}/, saved {model_name}-compare.pdf to plots/results/")
        print(f"Note: we recommend creating plots/{run_name}/ to store this plot")


def hyperparameter_eval(event_detector, model_type, config, val_errors, test_errors, Ytest,
    eval_metrics=['F1'],
    cutoffs=['0.995'],
    windows=[1],
    run_name='results'):

    model_name = config['name']
    do_batches = False
    all_Yhats = []

    for metric in eval_metrics:

        # FP is a negative metric (lower is better)
        negative_metric = (metric == 'FP')

        # FP is a negative metric (lower is better)
        if negative_metric:
            best_metric = 1
        else:
            best_metric = -1000

        best_metric = -1000
        best_percentile = 0
        best_window = 0
        metric_vals = np.zeros((len(cutoffs), len(windows)))
        metric_func = metrics.get(metric)

        for percentile_idx in range(len(cutoffs)):

            percentile = cutoffs[percentile_idx]

            # set threshold as quantile of average reconstruction error
            theta = np.quantile(val_errors, percentile)

            for window_idx in range(len(windows)):

                window = windows[window_idx]

                # Need to truncate on inner loop, since window values change
                Yhat = event_detector.cached_detect(test_errors, theta = theta, window = window)
                print("Yhat1", Yhat)
                Yhat = Yhat[window-1:].astype(int)
                print("Yhat2", Yhat)
                Yhat_trunc, Ytest_trunc = utils.normalize_array_length(Yhat, Ytest)
                choice_value = metric_func(Yhat_trunc, Ytest_trunc)
                metric_vals[percentile_idx, window_idx] = choice_value

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

        print("Best metric ({}) is {:.3f} at percentile={:.5f}, window {}".format(metric, best_metric, best_percentile, best_window))

        best_theta = np.quantile(val_errors, best_percentile)
        final_Yhat = event_detector.cached_detect(test_errors, theta = best_theta, window = best_window)
        final_Yhat = final_Yhat[best_window-1:].astype(int)

        metric_func = metrics.get(metric)

        final_Yhat_trunc, Ytest_trunc = utils.normalize_array_length(final_Yhat, Ytest)
        final_value = metric_func(final_Yhat_trunc, Ytest_trunc)

        try:
            np.save(f'outputs/{run_name}/{model_name}-{metric}.npy', metric_vals)
            print(f'Saved {model_name}-{metric}.npy to outputs/{run_name}/')
        except FileNotFoundError:
            np.save(f'outputs/results/{model_name}-{metric}.npy', metric_vals)
            print(f"Unable to find outputs/{run_name}/, saved {model_name}-{metric}.npy to outputs/results/")
            print(f"Note: we recommend creating outputs/{run_name}/ to store this output")

        print("Final {} is {:.3f} at percentile={:.5f}, window {}".format(metric, final_value, best_percentile, best_window))
        config['eval'].append({'percentile': best_percentile, 'window': best_window})

    return best_percentile, best_window

def parse_arguments():

    parser = utils.get_argparser()

    parser.add_argument("--detect_params_metrics",
            default=['F1'],
            nargs='+',
            type=str,
            help="Metrics to look over")

    # Detection hyperparameter search
    parser.add_argument("--detect_params_percentile", 
        default=[0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.99995],
        nargs='+',
        type=float,
        help="Percentiles to look over")

    parser.add_argument("--detect_params_windows", 
        default=[1, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        nargs='+',
        type=int,
        help="Windows to look over")

    parser.add_argument("--eval_plots",
        action='store_true',
        help="Make detection plots for provided hyperparameter settings")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    model_type = args.model
    dataset_name = args.dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

    # Generic evaluation settings
    config = {
        'eval': []
    }

    run_name = args.run_name
    utils.update_config_model(args, config, model_type, dataset_name)

    model_name = config['name']
    Xfull, sensor_cols = load_train_data(dataset_name, train_shuffle=True)
    Xtest, Ytest, _ = load_test_data(dataset_name)

    event_detector = load_saved_model(model_type, run_name, model_name)
    do_batches = False

    print('Getting detection errors....')

    if not model_type == 'AE':

        # Clip the prediction to match prediction window
        history = config['model']['history']
        Ytest = Ytest[history + 1:]
        do_batches = True

        all_idxs = np.arange(history, len(Xfull)-1)
        _, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True)
        validation_errors = utils.reconstruction_errors_by_idxs(event_detector, Xfull, val_idxs, history)

    else:

        _, Xval, _, _  = train_test_split(Xfull, Xfull, test_size=0.2, random_state=42, shuffle=True)
        validation_errors = event_detector.reconstruction_errors(Xval, batches=do_batches)

    test_errors = event_detector.reconstruction_errors(Xtest, batches=do_batches)

    validation_instance_errors = validation_errors.mean(axis=1)
    test_instance_errors = test_errors.mean(axis=1)

    bestp, bestw = hyperparameter_eval(event_detector,
        model_type,
        config,
        validation_instance_errors,
        test_instance_errors,
        Ytest,
        eval_metrics=args.detect_params_metrics,
        cutoffs=args.detect_params_percentile,
        windows=args.detect_params_windows,
        run_name=run_name)

    if args.eval_plots:

        # Makes a plot of the "best" setting. Additional settings can be included in the config.
        eval_demo(event_detector,
            model_type,
            config,
            validation_instance_errors,
            test_instance_errors,
            Ytest,
            eval_metrics=args.detect_params_metrics,
            run_name=run_name,
            include_best=True)

    print("Finished!")