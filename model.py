import numpy as np
import tensorflow as tf
from keras.layers import Permute, Dense, TimeDistributed, Conv2D, MaxPooling2D, Multiply, Dropout, Flatten, Reshape, \
    LSTM, UpSampling2D, concatenate
from keras import Input, Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import plot_model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(3, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def decoder(inputs, name):
    decode = Dense(dim)(inputs)
    decode = Dropout(0.3)(decode)
    decode = Reshape((4, 4, 16))(decode)
    decode = UpSampling2D((2, 2))(decode)
    decode = Conv2D(8, (3, 3), padding='same')(decode)
    decode = UpSampling2D((2, 2))(decode)
    decoder_output = Conv2D(1, (3, 3), padding='same', name=name+'_output')(decode)
    return decoder_output


[demandX_train, supplyX_train] = np.load('train.npz')['X']
[demandY_train, supplyY_train] = np.load('train.npz')['Y']
[demandX_test, supplyX_test] = np.load('test.npz')['X']
[demandY_test, supplyY_test] = np.load('test.npz')['Y']


timestep = 3
size = 16
dim = 4 * 4 * 16

input_demand = Input(shape=(None, size, size, 1))
demand_cnn = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(input_demand)
demand_cnn = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(demand_cnn)
demand_cnn = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(demand_cnn)
demand_cnn = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(demand_cnn)
demand_cnn = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(demand_cnn)
demand_cnn = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(demand_cnn)
demand_cnn = TimeDistributed(Dropout(0.3))(demand_cnn)
demand_cnn = TimeDistributed(Flatten())(demand_cnn)
demand_cnn = Reshape((timestep, dim))(demand_cnn)

input_supply = Input(shape=(None, size, size, 1))
supply_cnn = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(input_supply)
supply_cnn = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(supply_cnn)
supply_cnn = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(supply_cnn)
supply_cnn = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(supply_cnn)
supply_cnn = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(supply_cnn)
supply_cnn = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(supply_cnn)
supply_cnn = TimeDistributed(Dropout(0.3))(supply_cnn)
supply_cnn = TimeDistributed(Flatten())(supply_cnn)
supply_cnn = Reshape((timestep, dim))(supply_cnn)

combine_demand_supply = concatenate([demand_cnn, supply_cnn])

attention = LSTM(dim, return_sequences=1, input_shape=(timestep, dim * 2))(combine_demand_supply)
attention = attention_3d_block(attention)
attention = Flatten()(attention)
attention = Dense(dim * 2)(attention)




demand_decoder = decoder(attention, 'demand')
supply_decoder = decoder(attention, 'supply')


model = Model(inputs=[input_demand, input_supply], outputs=[demand_decoder, supply_decoder])
model.compile(loss='mse', optimizer='adadelta', metrics=[rmse])

print(model.summary())


plot_model(model, to_file='model.png')

history = model.fit([demandX_train, supplyX_train], [demandY_train, supplyY_train],
                    batch_size=16,
                    epochs=70,
                    verbose=2,
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
plt.axhline(2.8)
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
