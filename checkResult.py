from scipy.io import wavfile
import numpy as np

def pre_emphasis_filter(x, coeff=0.95):
    return np.concatenate([x, np.subtract(x, np.multiply(x, coeff))])

true_rate, y_true = wavfile.read("models/memOpt/y_test.wav")
pred_rate, y_pred = wavfile.read("models/memOpt/y_pred.wav")

print(len(y_true))
print(len(y_pred))

y_true, y_pred = pre_emphasis_filter(y_true), pre_emphasis_filter(y_pred)

print(len(y_true))
print(len(y_pred))