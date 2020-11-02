import numpy as np
GCN_X = np.load("train.npz")["X"][1][:, 0].reshape([1536,256,1])
GCN_Y = np.load("train.npz")["Y"][1].reshape([1536,256,1])
print(GCN_X.shape)
print(GCN_Y.shape)