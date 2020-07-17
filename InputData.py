import numpy as np


def InputDataReshape(data, factor=0):
    firstweek = 96 * 7
    group_num = len(data) - firstweek

    #factorX = []
    quarterX = []
    dayX = []
    weekX = []
    Y = []
    for i in range(group_num):
        Y.append(data[firstweek + i])
        weekX.append(data[i])
        dayX.append(data[firstweek + i - 96])
        quarterX.append(data[firstweek + i - 1])
        #factorX.append(factor[firstweek + i])

    #factorX = np.asarray(factorX)
    weekX = np.asarray(weekX).reshape((-1, 1, 16, 16, 1))
    dayX = np.asarray(dayX).reshape((-1, 1, 16, 16, 1))
    quarterX = np.asarray(quarterX).reshape((-1, 1, 16, 16, 1))
    Y = np.asarray(Y)
    return np.concatenate([weekX, dayX, quarterX], axis=1), Y


data = np.load('CNN_input_allDIDI.npz')['arr_0']
demand = data[:, :, :, :1]
supply = data[:, :, :, 1:]

demandX, demandY = InputDataReshape(demand)
supplyX, supplyY = InputDataReshape(supply)

test = 96 * 7
size = 16


demandX_train, demandY_train, supplyX_train, supplyY_train = \
    demandX[:-test], demandY[:-test], supplyX[:-test], supplyY[:-test]

np.savez('train',
         X=[demandX_train, supplyX_train],
         Y=[demandY_train, supplyY_train])

demandX_test, demandY_test, supplyX_test, supplyY_test = \
    demandX[-test:], demandY[-test:], supplyX[-test:], supplyY[-test:]

np.savez('test',
         X=[demandX_test, supplyX_test],
         Y=[demandY_test, supplyY_test])
