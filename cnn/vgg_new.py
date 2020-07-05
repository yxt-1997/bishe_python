# -*- coding: utf-8 -*-
######仿照VGG16的网络1
import numpy as np
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.optimizers import adam
import scipy.io as sio
import  os
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
path2='E:/毕设/代码/407test/cnn/cnn_train_pinjie.mat'
test = sio.loadmat(path2)
test2=test['cnn_train_pinjie']
x_all1=test2[12000:, 0:3072]
y_all1=test2[12000:,-1]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_all1, y_all1, random_state=1, test_size=24000)
nb_features = 1024
# x_train = np.array(x_train)
# x_test = np.array(x_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# X_train_r = np.zeros((len(x_train), nb_features, 2))
# X_train_r[:, :, 0] = x_train[:, :nb_features]#取X_train 的前64列
# X_train_r[:, :, 1] = x_train[:, :nb_features]
# X_test_r = np.zeros((len(x_test), nb_features, 2))
# X_test_r[:, :, 0] = x_test[:, :nb_features]#取X_train 的前64列
# X_test_r[:, :, 1] = x_test[:, nb_features:2048]
X_train_r = np.zeros((len(x_train), nb_features, 3))  # 3通道的输入
X_train_r[:, :, 0] = x_train[:, :nb_features]  # 取X_train 的前1024列
X_train_r[:, :, 1] = x_train[:, nb_features:2048]
X_train_r[:, :, 2] = x_train[:, 2048:]
# reshape validation data
X_test_r = np.zeros((len(x_test), nb_features, 3))
X_test_r[:, :, 0] = x_test[:, :nb_features]
X_test_r[:, :, 1] = x_test[:, nb_features:2048]
X_test_r[:, :, 2] = x_test[:, 2048:]
nb_class = 6
model = Sequential()
model.add(Convolution1D(nb_filter=8, filter_length=3, strides=1,input_shape=(nb_features, 3), padding="same"))
model.add(Activation('relu'))
model.add(Convolution1D(nb_filter=8, filter_length=3, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Convolution1D(nb_filter=16, filter_length=3,strides=1, padding="same"))
model.add(Activation('relu'))
model.add(Convolution1D(nb_filter=16, filter_length=3, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Convolution1D(nb_filter=32, filter_length=3,strides=1, padding="same"))
model.add(Activation('relu'))
model.add(Convolution1D(nb_filter=32, filter_length=3, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Convolution1D(nb_filter=64, filter_length=3,strides=1, padding="same"))
model.add(Activation('relu'))
model.add(Convolution1D(nb_filter=64, filter_length=3, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Convolution1D(nb_filter=128, filter_length=3,strides=2, padding="same"))
model.add(Activation('relu'))
model.add(Convolution1D(nb_filter=128, filter_length=3, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Convolution1D(nb_filter=256, filter_length=3, padding="same"))
model.add(Activation('relu'))
model.add(Convolution1D(nb_filter=256, filter_length=3, padding="same"))
model.add(Activation('relu'))
model.add(Convolution1D(nb_filter=256, filter_length=3, padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
# model.add(Dense(1024, activation='relu'))
model.add(Dense(6))
model.add(Activation('softmax'))
y_train = np_utils.to_categorical(y_train, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)
adam = adam(lr=1e-4)
sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
nb_epoch = 50
history=model.fit(X_train_r, y_train, nb_epoch=40, validation_data=(X_test_r, y_test), batch_size=256,verbose=2,callbacks=[early_stopping])
result = model.predict(X_test_r)
model.save('vgg4181.h5')
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model, to_file='vgg.png')