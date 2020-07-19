import numpy as np
import tensorflow as tf
from keras.layers import Permute, Dense, TimeDistributed, Conv2D, MaxPooling2D, Multiply, Dropout, Flatten, Reshape, \
    LSTM, UpSampling2D, concatenate
from keras import Input, Model
from keras import backend as K
import matplotlib.pyplot as plt


# from keras.utils import plot_model


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
    decoder_output = Conv2D(1, (3, 3), padding='same', name=name + '_output')(decode)
    return decoder_output


def encoder(inputs):
    encode = TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='relu'))(inputs)
    encode = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(encode)
    encode = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(encode)
    encode = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(encode)
    encode = TimeDistributed(Dropout(0.3))(encode)
    encode = TimeDistributed(Flatten())(encode)
    encode = Reshape((timestep, dim))(encode)
    return encode


[demandX_train, supplyX_train] = np.load('train.npz')['X']
[demandY_train, supplyY_train] = np.load('train.npz')['Y']
factor_train = np.load('train.npz')['factor']
demand_aux_train = np.load('train.npz')['auxiliary'][:, :, :, :1]
supply_aux_train = np.load('train.npz')['auxiliary'][:, :, :, 1:]
[demandX_test, supplyX_test] = np.load('test.npz')['X']
[demandY_test, supplyY_test] = np.load('test.npz')['Y']
factor_test = np.load('test.npz')['factor']
demand_aux_test = np.load('test.npz')['auxiliary'][:, :, :, :1]
supply_aux_test = np.load('test.npz')['auxiliary'][:, :, :, 1:]

timestep = 3
size = 16
dim = 4 * 4 * 16

input_demand = Input(shape=(None, size, size, 1))
demand_encoder = encoder(input_demand)

input_supply = Input(shape=(None, size, size, 1))
supply_encoder = encoder(input_supply)

combine_demand_supply = concatenate([demand_encoder, supply_encoder])
lstm = LSTM(dim, return_sequences=1, input_shape=(timestep, dim * 2))(combine_demand_supply)

input_aux = Input(shape=(size, size, 12))
aux_encode = Conv2D(8, (3, 3), padding='same', activation='relu')(input_aux)
aux_encode = MaxPooling2D(pool_size=(2, 2))(aux_encode)
aux_encode = Conv2D(16, (3, 3), padding='same', activation='relu')(aux_encode)
aux_encode = MaxPooling2D(pool_size=(2, 2))(aux_encode)
aux_decode = Conv2D(16, (3, 3), padding='same', activation='relu')(aux_encode)
aux_decode = UpSampling2D((2, 2))(aux_decode)
aux_decode = Conv2D(8, (3, 3), padding='same', activation='relu')(aux_decode)
aux_decode = UpSampling2D((2, 2))(aux_decode)
aux_decode = Conv2D(12, (3, 3), padding='same', activation='relu')(aux_decode)


def aux_task(inputs):
    aux_predict = Reshape((4, 4, 16))(inputs)
    aux_predict = Conv2D(16, (3, 3), padding='same', activation='relu')(aux_predict)
    aux_predict = UpSampling2D((2, 2))(aux_predict)
    aux_predict = Conv2D(8, (3, 3), padding='same', activation='relu')(aux_predict)
    aux_predict = UpSampling2D((2, 2))(aux_predict)
    aux_predict = Conv2D(1, (3, 3), padding='same', activation='relu')(aux_predict)
    return aux_predict


aux_dim = 16*4*4
aux = Reshape((aux_dim,))(aux_encode)

aux_demand = Dense(aux_dim)(aux)
aux_demand = Dense(aux_dim)(aux_demand)
aux_demand_predict = aux_task(aux_demand)

aux_supply = Dense(aux_dim)(aux)
aux_supply = Dense(aux_dim)(aux_supply)
aux_supply_predict = aux_task(aux_supply)


demand_attention = attention_3d_block(lstm)
demand_attention = Flatten()(demand_attention)
demand_attention = Dense(dim * 2)(demand_attention)
demand_combine = concatenate([demand_attention, aux_demand])
demand_combine = Dense(dim * 2)(demand_combine)
demand_decoder = decoder(demand_combine, 'demand')

supply_attention = attention_3d_block(lstm)
supply_attention = Flatten()(supply_attention)
supply_attention = Dense(dim * 2)(supply_attention)
supply_combine = concatenate([supply_attention, aux_supply])
supply_combine = Dense(dim * 2)(supply_combine)
supply_decoder = decoder(supply_combine, 'supply')

model = Model(inputs=[input_demand, input_supply, input_aux],
              outputs=[demand_decoder, supply_decoder, aux_decode, aux_demand_predict, aux_supply_predict])
model.compile(loss='mse', optimizer='adadelta', metrics=[rmse])

print(model.summary())
# plot_model(model, to_file='model.png')

history = model.fit([demandX_train, supplyX_train, factor_train],
                    [demandY_train, supplyY_train, factor_train, demand_aux_train, supply_aux_train],
                    batch_size=8,
                    epochs=300,
                    verbose=2,
                    validation_data=([demandX_test, supplyX_test, factor_test],
                                     [demandY_test, supplyY_test, factor_test, demand_aux_test, supply_aux_test]))

demand_rmse = history.history['demand_output_rmse']
val_demand_rmse = history.history['val_demand_output_rmse']
supply_rmse = history.history['supply_output_rmse']
val_supply_rmse = history.history['val_supply_output_rmse']

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
