import numpy as np

pois = np.load('CNN_DIDI_withPOI.npz')['arr_0']
fac = np.load('LSTM.npz')['arr_0'][:, :5]
fac = np.reshape(fac, (2880, 1, 1, 5)).astype(np.float)


poi = pois[:, :, :, 2:]

c = np.zeros((2880, 16, 16, 5), dtype=float)
d = fac + c

eventfile = np.load('LSTM_event.npz')['arr_0']
event = np.zeros((2880, 16, 16, 1))
for timeslice in eventfile:
    if timeslice[1] == -1:
        continue
    else:
        event[timeslice[0]][timeslice[1]][timeslice[2]][0] = 1

factor = np.concatenate([poi, d, event], axis=3)

print(factor.shape)
np.save('factor', factor)
