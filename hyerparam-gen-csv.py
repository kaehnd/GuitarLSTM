import tensorflow as tf
import keras
import keras.layers as layers
from keras import Sequential, Input
from keras.layers import LSTM, Conv1D, Dense
from keras.backend import clear_session
from keras.activations import tanh, elu, relu
from keras.models import load_model
import keras.backend as K
import keras_tuner as kt

from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import argparse
import time
import json
import csv

class GuitarLSTMModel(kt.HyperModel):
    def build(self, hp):
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        conv1d_strides = hp.Int('conv1d_strides', 3, 12)
        conv1d_filters = hp.Int('conv1d_filters', 12, 26)
        hidden_units= hp.Int('hidden_units', 36, 126)
        input_size = hp.Int('input_size', 50, 160)
        
        model = Sequential()
        model.add(Conv1D(conv1d_filters, 12,strides=conv1d_strides, activation=None, padding='same',input_shape=(input_size,1)))
        model.add(Conv1D(conv1d_filters, 12,strides=conv1d_strides, activation=None, padding='same'))
        model.add(LSTM(hidden_units))
        model.add(Dense(1, activation=None))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=error_to_signal, metrics=[error_to_signal])
        return model

    def fit(self, hp : kt.HyperParameters, model, *args, **kwargs):
        input_size = hp.get('input_size')
        X_all = args[0]
        y_all = args[1]

        y_ordered = y_all[input_size-1:] 

        indices = np.arange(input_size) + np.arange(len(X_all)-input_size+1)[:,np.newaxis] 
        X_ordered = tf.gather(X_all,indices) 

        shuffled_indices = np.random.permutation(len(X_ordered)) 
        X_random = tf.gather(X_ordered,shuffled_indices)
        y_random = tf.gather(y_ordered, shuffled_indices)
        return model.fit(
            X_random,
            y_random,
            **kwargs,
        )

def pre_emphasis_filter(x, coeff=0.95):
    return tf.concat([x, x - coeff * x], 1)
    
def error_to_signal(y_true, y_pred): 
    """
    Error to signal ratio with pre-emphasis filter:
    """
    y_true, y_pred = pre_emphasis_filter(y_true), pre_emphasis_filter(y_pred)
    return K.sum(tf.pow(y_true - y_pred, 2), axis=0) / K.sum(tf.pow(y_true, 2), axis=0) + 1e-10
    
def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float16))

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm


def get_metric_value(metrics, value):
    return metrics[value].get_history()[0].value[0]

def main():
    '''Ths is a similar Tensorflow/Keras implementation of the LSTM model from the paper:
        "Real-Time Guitar Amplifier Emulation with Deep Learning"
        https://www.mdpi.com/2076-3417/10/3/766/htm

        Uses a stack of two 1-D Convolutional layers, followed by LSTM, followed by 
        a Dense (fully connected) layer. Three preset training modes are available, 
        with further customization by editing the code. A Sequential tf.keras model 
        is implemented here.
    '''

    print('about to build tuner')

    tuner = kt.Hyperband(
        GuitarLSTMModel(),
        objective=kt.Objective('val_error_to_signal', direction='min'),
        max_epochs=8,
        hyperband_iterations=30
      )
   

    print('about to reload tuner')

    tuner.reload()

    times = {}
    # for trial in tuner.oracle.trials:
    trials = list(tuner.oracle.trials.values())

    with open('times.json', 'r') as jsonFile:
        times = json.load(jsonFile)


    with open('data.csv', 'w', newline='') as f:

        header = [
            'trial_id', 
            'learning_rate', 
            'conv1d_strides',
            'conv1d_filters',
            'hidden_units',
            'input_size',
            'tuner/epochs',
            'tuner/initial_epoch',
            'tuner/bracket',
            'tuner/round',
            'loss',
            'error_to_signal',
            'val_loss',
            'val_error_to_signal',
            'prediction_time'
        ]

        writer = csv.writer(f)
        writer.writerow(header)

        for trial in trials :
            print('\tTrial #: ' + trial.trial_id)

            hp : kt.HyperParameters = trial.hyperparameters
            metrics = trial.metrics.metrics

            writer.writerow([
                trial.trial_id,
                hp.get('learning_rate'),
                hp.get('conv1d_strides'),
                hp.get('conv1d_filters'),
                hp.get('hidden_units'),
                hp.get('input_size'),
                hp.get('tuner/epochs'),
                hp.get('tuner/initial_epoch'),
                hp.get('tuner/bracket'),
                hp.get('tuner/round'),
                get_metric_value(metrics, 'loss'),
                get_metric_value(metrics, 'error_to_signal'),
                get_metric_value(metrics, 'val_loss'),
                get_metric_value(metrics, 'val_error_to_signal'),
                times[trial.trial_id]
            ])



if __name__ == "__main__":
    main()
