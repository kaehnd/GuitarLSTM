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

class GuitarLSTMModel(kt.HyperModel):
    def build(self, hp):
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        conv1d_strides = hp.Int('conv1d_strides', 3, 12)
        conv1d_filters = hp.Int('conv1d_filters', 12, 26)
        hidden_units= hp.Int('hidden_units', 36, 126)
        input_size = hp.Int('input_size', 50, 200)
        
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
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm

def main(args):
    '''Ths is a similar Tensorflow/Keras implementation of the LSTM model from the paper:
        "Real-Time Guitar Amplifier Emulation with Deep Learning"
        https://www.mdpi.com/2076-3417/10/3/766/htm

        Uses a stack of two 1-D Convolutional layers, followed by LSTM, followed by 
        a Dense (fully connected) layer. Three preset training modes are available, 
        with further customization by editing the code. A Sequential tf.keras model 
        is implemented here.
    '''

    name = args.name

    batch_size = args.batch_size 
    test_size = 0.2

    clear_session()

    tuner = kt.Hyperband(
        GuitarLSTMModel(),
        objective="mean_squared_error",
        max_epochs=args.max_epochs,
        hyperband_iterations=30
    )

    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)

    X_all = in_data.astype(np.float32).flatten()  
    X_all = normalize(X_all).reshape(len(X_all),1)   
    y_all = out_data.astype(np.float32).flatten() 
    y_all = normalize(y_all).reshape(len(y_all),1)   

    # Run optimization ###################################################
    tuner.search(X_all,y_all, epochs=30, batch_size=batch_size, validation_split=test_size)    

    models = tuner.get_best_models(num_models=1)
    hps = tuner.get_best_hyperparameters(num_trials=1)

    model = models[0]
    hp = hps[0]

    model.save('models/'+name+'/'+name+'.h5')

    input_size = hp.get('input_size')

    # Get the last 20% of the wav data to run prediction and plot results
    y_the_rest, y_last_part = np.split(y_all, [int(len(y_all)*.8)])
    x_the_rest, x_last_part = np.split(X_all, [int(len(X_all)*.8)])
    y_test = y_last_part[input_size-1:] 
    indices = np.arange(input_size) + np.arange(len(x_last_part)-input_size+1)[:,np.newaxis] 
    X_test = tf.gather(x_last_part,indices) 

    prediction = model.predict(X_test, batch_size=batch_size)

    save_wav('models/'+name+'/y_pred.wav', prediction)
    save_wav('models/'+name+'/x_test.wav', x_last_part)
    save_wav('models/'+name+'/y_test.wav', y_test)

    # Add additional data to the saved model (like input_size)
    filename = 'models/'+name+'/'+name+'.h5'
    f = h5py.File(filename, 'a')
    grp = f.create_group("info")
    dset = grp.create_dataset("input_size", (1,), dtype='int16')
    dset[0] = input_size
    f.close()

    # Create Analysis Plots ###########################################
    if args.create_plots == 1:
        print("Plotting results..")
        import plot

        plot.analyze_pred_vs_actual({   'output_wav':'models/'+name+'/y_test.wav',
                                            'pred_wav':'models/'+name+'/y_pred.wav', 
                                            'input_wav':'models/'+name+'/x_test.wav',
                                            'model_name':name,
                                            'show_plots':1,
                                            'path':'models/'+name
                                        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("name")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_epochs", type=int, default=8)
    parser.add_argument("--create_plots", type=int, default=0)

    args = parser.parse_args()
    main(args)
