import numpy as np

a = np.load('CNN_DIDI_withPOI.npz')['arr_0']
b = np.load('LSTM.npz')['arr_0'][:, :6]
b = np.reshape(b, (2880, 1, 1, 6)).astype(np.float)

poi = a[:, :, :, 2:]

c = np.zeros((2880, 16, 16, 6), dtype=float)
d = b + c

factor = np.concatenate([poi, d], axis=3)

np.save('factor', factor)
