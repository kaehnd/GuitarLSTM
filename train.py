import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence

import os
import math
from scipy.io import wavfile
import numpy as np
import h5py
import argparse
import plot

class WindowArray(Sequence):
    def __init__(self, x, y, window_len, batch_size=32):
        self.x = x
        self.y = y[window_len:] 
        self.window_len = window_len
        self.batch_size = batch_size
        
    def __len__(self):
        return math.ceil((len(self.x) - self.window_len) / self.batch_size)
    
    def __getitem__(self, index):
        end_index = (index+1) * self.batch_size
        if index == self.__len__() - 1:
            end_index = len(self.x) - self.window_len

        stack = []
        for idx in range(index * self.batch_size, end_index):
            stack.append(self.x[idx:idx+self.window_len])
        x_out = np.stack(stack)
        y_out = self.y[index * self.batch_size:end_index]
        return x_out, y_out
   
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

        Note: RAM may be a limiting factor for the parameter "input_size". The wav data
          is preprocessed and stored in RAM, which improves training speed but quickly runs out
          if using a large number for "input_size".  Reduce this if you are experiencing
          RAM issues.
        
        --training_mode=0   Speed training (default)
        --training_mode=1   Accuracy training
        --training_mode=2   Extended training (set max_epochs as desired, for example 50+)
    '''

    name = args.name
    if not os.path.exists('models/'+name):
        os.makedirs('models/'+name)
    else:
        print("A model folder with the same name already exists. Please choose a new name.")
        return

    train_mode = args.training_mode     # 0 = speed training
                                        # 1 = accuracy training
                                        # 2 = extended training
    batch_size = args.batch_size
    epochs = args.max_epochs
    input_size = args.input_size

    # TRAINING MODE
    if train_mode == 0:         # Speed Training
        learning_rate = 0.01 
        conv1d_strides = 12    
        conv1d_filters = 16
        hidden_units = 36
    elif train_mode == 1:       # Accuracy Training (~10x longer than Speed Training)
        learning_rate = 0.01 
        conv1d_strides = 4
        conv1d_filters = 36
        hidden_units= 64
    elif train_mode == 2:       # Extended Training (~60x longer than Accuracy Training)
        learning_rate = 0.0005 
        conv1d_strides = 3
        conv1d_filters = 36
        hidden_units= 96
    else:                       # Optimal
        learning_rate = 0.00756 
        conv1d_strides = 8
        conv1d_filters = 12
        hidden_units= 107

    # Create Sequential Model ###########################################
    clear_session()
    model = Sequential()
    model.add(Conv1D(conv1d_filters, 12,strides=conv1d_strides, activation=None, padding='same',input_shape=(input_size,1)))
    model.add(Conv1D(conv1d_filters, 12,strides=conv1d_strides, activation=None, padding='same'))
    # model.add(LSTM(hidden_units))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation=None))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[error_to_signal])
    print(model.summary())

    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)

    X_all = in_data.astype(np.float16).flatten()
    X_all = normalize(X_all).reshape(len(X_all),1)
    y_all = out_data.astype(np.float16).flatten()
    y_all = normalize(y_all).reshape(len(y_all),1)

    train_examples = int(len(X_all)*0.8)
    train_arr = WindowArray(X_all[:train_examples], y_all[:train_examples], input_size, batch_size=batch_size)
    val_arr = WindowArray(X_all[train_examples:], y_all[train_examples:], input_size, batch_size=batch_size)

    # Train Model ###################################################
    model.fit(train_arr, validation_data=val_arr, epochs=epochs, shuffle=True)
    model.save('models/'+name+'/'+name+'.h5')

    # Run Prediction #################################################
    print("Running prediction..")

    # Get the last 20% of the wav data to run prediction and plot results
    y_the_rest, y_last_part = np.split(y_all, [int(len(y_all)*.8)])
    x_the_rest, x_last_part = np.split(X_all, [int(len(X_all)*.8)])
    y_test = y_last_part[input_size:]
    test_arr = WindowArray(x_last_part, y_last_part, input_size, batch_size=batch_size)

    prediction = model.predict(test_arr)

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
    parser.add_argument("--training_mode", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_epochs", type=int, default=8)
    parser.add_argument("--create_plots", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=89)
    parser.add_argument("--split_data", type=int, default=1)
    args = parser.parse_args()
    main(args)