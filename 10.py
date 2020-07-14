import numpy as np
from keras.layers import Permute, Dense, Multiply, TimeDistributed, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, LSTM, UpSampling2D, concatenate
from keras import Input, Model
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


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


def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(3, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


data = np.load('CNN_input_allDIDI.npz')['arr_0']
demand = data[:, :, :, :1]
supply = data[:, :, :, 1:]

demandX, demandY = InputDataReshape(demand)
supplyX, supplyY = InputDataReshape(supply)

timestep = 3
test = 96*7
size = 16
dim = 4 * 4 * 16

demandX_train, demandY_train, supplyX_train, supplyY_train = \
    demandX[:-test], demandY[:-test], supplyX[:-test], supplyY[:-test]

demandX_test, demandY_test, supplyX_test, supplyY_test = \
    demandX[-test:], demandY[-test:], supplyX[-test:], supplyY[-test:]


input_demand = Input(shape=(None, size, size, 1))
demand_cnn = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(input_demand)
#demand_cnn = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(demand_cnn)
demand_cnn = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(demand_cnn)
demand_cnn = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(demand_cnn)
demand_cnn = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(demand_cnn)
demand_cnn = TimeDistributed(Dropout(0.3))(demand_cnn)
demand_cnn = TimeDistributed(Flatten())(demand_cnn)
demand_cnn = Reshape((timestep, dim))(demand_cnn)

input_supply = Input(shape=(None, size, size, 1))
supply_cnn = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(input_supply)
#supply_cnn = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(supply_cnn)
supply_cnn = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(supply_cnn)
supply_cnn = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(supply_cnn)
supply_cnn = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(supply_cnn)
supply_cnn = TimeDistributed(Dropout(0.3))(supply_cnn)
supply_cnn = TimeDistributed(Flatten())(supply_cnn)
supply_cnn = Reshape((timestep, dim))(supply_cnn)

combine_demand_supply = concatenate([demand_cnn, supply_cnn])

attention = LSTM(dim, return_sequences=1, input_shape=(timestep, dim*2))(combine_demand_supply)
attention = attention_3d_block(attention)
attention = Flatten()(attention)
attention = Dense(dim*2)(attention)

output_demand = Dense(dim)(attention)
output_demand = Dropout(0.3)(output_demand)
output_demand = Reshape((4, 4, 16))(output_demand)
output_demand = UpSampling2D((2, 2))(output_demand)
output_demand = Conv2D(8, (3, 3), padding='same')(output_demand)
output_demand = UpSampling2D((2, 2))(output_demand)
output_demand = Conv2D(1, (3, 3), padding='same', name='demand')(output_demand)

output_supply = Dense(dim)(attention)
output_supply = Dropout(0.3)(output_supply)
output_supply = Reshape((4, 4, 16))(output_supply)
output_supply = UpSampling2D((2, 2))(output_supply)
output_supply = Conv2D(8, (3, 3), padding='same')(output_supply)
output_supply = UpSampling2D((2, 2))(output_supply)
output_supply = Conv2D(1, (3, 3), padding='same', name='supply')(output_supply)


model = Model(inputs=[input_demand, input_supply], outputs=[output_demand, output_supply])
model.compile(loss='mse', optimizer='adadelta', metrics=[rmse])

print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model.png')

history = model.fit([demandX_train, supplyX_train], [demandY_train, supplyY_train], batch_size=8, epochs=100,
                    validation_data=([demandX_test, supplyX_test], [demandY_test, supplyY_test]))


demand_rmse = history.history['demand_rmse']
val_demand_rmse = history.history['val_demand_rmse']
supply_rmse = history.history['supply_rmse']
val_supply_rmse = history.history['val_supply_rmse']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, demand_rmse, 'bo', label='Demand Training RMSE')
plt.plot(epochs, val_demand_rmse, 'b', label='Demand Validation RMSE')
plt.plot(epochs, supply_rmse, 'ro', label='Supply Training RMSE')
plt.plot(epochs, val_supply_rmse, 'r', label='Supply Validation RMSE')
plt.title('Training and validation RMSE')
plt.grid(1)
plt.axhline(3.8)
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

