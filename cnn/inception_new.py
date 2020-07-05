#####inception网络_按网络####
from keras.models import Model
from keras.layers import (Input,Activation,Dense,Flatten,Reshape,Concatenate,Dropout)
from keras.layers.convolutional import (Conv2D,MaxPooling2D,AveragePooling2D)
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import initializers
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import load_model
import csv
from scipy.io import loadmat
import scipy.io as sio


data=[]
x1=[]
y1=[]
channel=3
nb_features=1024
class_num=6
nb_class = 6
momentum = 0.85
learning_rate = 0.01
epoch=100
path2='E:/毕设/代码/331_cnn/cnn_train_pinjie.mat'
path2='E:/毕设/代码/407test/cnn/cnn_train_pinjie.mat'
test = sio.loadmat(path2)
test2=test['cnn_train_pinjie']
x_all1=test2[12000:, 0:3072]
y_all1=test2[12000:,-1]
# path3='E:/毕设/代码/cnn/alldata_cnn1.mat'
# test3 = sio.loadmat(path3)
# test3=test3['alldata_cnn']
# test3=test3[:,:,30]
# x_all=test3[:,0:3072]
# y_all=test3[:,-1]
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_all1, y_all1, random_state=1, test_size=24000)
#验证样本数4000
print("x_train shape",x_train.shape)
print("y_train.shape",y_train.shape)
x_train=x_train.reshape(-1,1024,3)
x_val=x_val.reshape(-1,1024,3)
y_train = np_utils.to_categorical(y_train, nb_class)#转换成独热码
y_val= np_utils.to_categorical(y_val, nb_class)
nb_features = 1024

initializer=initializers.he_normal(seed=None)#定义权重初始化方式
filepath='inception_resnet300.h5'
batch_size=256
def inception_function(x_input, filter1, filter2):###inception 网络
    ###A表示第几层， a 表示第几分支####
    ####第一分支####
  conv_1x1_A_a = Conv2D(filters=filter1, kernel_size=(1, 1), strides=(1, 1), padding="same",
                      kernel_initializer=initializer)(x_input)
  conv_1x1_A_a = BatchNormalization(axis=-1, momentum=momentum)(conv_1x1_A_a)
  f_A_a = Activation('relu')(conv_1x1_A_a)
  conv_3x1_B_a = Conv2D(filters=filter2, kernel_size=(3, 1), strides=(1, 1), padding="same",
                      kernel_initializer=initializer)(f_A_a)
  conv_3x1_B_a = BatchNormalization(axis=-1, momentum=momentum)(conv_3x1_B_a)
  f_B_a = Activation('relu')(conv_3x1_B_a)
  print('分支1', f_B_a.shape)
   #####第二分支#######

  conv_1x1_A_b=Conv2D(filters=filter1, kernel_size=(1, 1), strides=(1, 1), padding="same",
                      kernel_initializer=initializer)(x_input)
  conv_1x1_A_b = BatchNormalization(axis=-1, momentum=momentum)(conv_1x1_A_b)
  f_A_b = Activation('relu')(conv_1x1_A_b)
  conv_5x1_B_b = Conv2D(filters=filter2, kernel_size=(5, 1), strides=(1, 1), padding="same",
                               kernel_initializer=initializer)(f_A_b)
  conv_5x1_B_b = BatchNormalization(axis=-1, momentum=momentum)(conv_5x1_B_b )
  f_B_b = Activation('relu')(conv_5x1_B_b)
  print('分支2', f_B_b.shape)
  #####第三分支#####

  conv_1x1_A_c=Conv2D(filters=filter2, kernel_size=(1, 1), strides=(1, 1), padding="same",
                      kernel_initializer=initializer)(x_input)
  conv_1x1_A_c= BatchNormalization(axis=-1, momentum=momentum)(conv_1x1_A_c)
  f_A_c = Activation('relu')(conv_1x1_A_c)
  print('分支2', f_A_c.shape)
  ######第四分支######

  max_3x1_A_d = MaxPooling2D((3, 1), strides=(1, 1), padding="same")(x_input)
  conv_1x1_B_d=Conv2D(filters=filter2, kernel_size=(1, 1), strides=(1, 1), padding="same",
                      kernel_initializer=initializer)(max_3x1_A_d)
  conv_1x1_B_d = BatchNormalization(axis=-1, momentum=momentum)(conv_1x1_B_d)
  f_B_d=Activation('relu')(conv_1x1_B_d)
  print('分支2', f_B_d.shape)
# Concatenate----------------------------------------------------------
  inceptionA = Concatenate(axis=3)([f_B_a, f_B_b, f_A_c, f_B_d])
  x1 = BatchNormalization(axis=-1, momentum=momentum)(inceptionA)
  print('合并', x1.shape)
  x1 = MaxPooling2D((3, 1), strides=(2, 1), padding="same")(x1)
  print('pool',x1.shape)
  return (x1)

##构建模型
input = Input(shape=(1024, 3))
input_reshape = Reshape((1024,1, 3))(input)
conv1 = Conv2D(filters=8, kernel_size=(5, 1), strides=(2, 1), padding="same", kernel_initializer=initializer)(input_reshape)
#conv1=BatchNormalization(axis=-1, momentum=momentum)(conv1)
x1 = Activation('relu')(conv1)
print('after conv1',conv1.shape)
x1 = BatchNormalization(axis=-1, momentum=momentum)(x1)
x1=MaxPooling2D((3, 1), strides=(2, 1), padding="same")(x1)

print('after maxpooling1',x1.shape)
conv2 = Conv2D(filters=16, kernel_size=(3, 1), strides=(2, 1), padding="same", kernel_initializer=initializer)(x1)
conv2=BatchNormalization(axis=-1, momentum=momentum)(conv2)
x2 = Activation('relu')(conv2)
print('after conv2',conv2.shape)
x4 = BatchNormalization(axis=-1, momentum=momentum)(x2)
x4=MaxPooling2D((3, 1), strides=(2, 1), padding="same")(x4)
print('after maxpooling2',x4.shape)
print('inception1----begin-----')
ince1=inception_function(x4, 3, 4)
print('after inception1',ince1.shape)
#x5=MaxPooling2D((3, 1), strides=(2, 1), padding="same")(ince1)
ince2=inception_function(ince1, 6, 8)
print('inception2----begin-----')
print('after inception2',ince2.shape)
#x6=MaxPooling2D((3, 1), strides=(2, 1), padding="same")(ince2)
ince3=inception_function(ince2, 6, 8)
print('inception3----begin-----')
print('after inception3',ince3.shape)
#x7=MaxPooling2D((3, 1), strides=(2, 1), padding="same")(ince3)
ince4=inception_function(ince3, 12, 16)
print('inception4----begin-----')
print('after inception4',ince4.shape)
#x8=MaxPooling2D((3, 1), strides=(2, 1), padding="same")(ince4)
x = AveragePooling2D((4, 1), strides=(4, 1), padding="same")(ince4)
print('after aver',x.shape)
flatten = Flatten(name='flatten')(x)
print('Flatten', flatten.shape)
predict = Dense(class_num, activation='softmax', name='fc', kernel_initializer='he_normal')(flatten)
print('predict', predict.shape)
model = Model(inputs=input, outputs=predict)
adam = optimizers.adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
check_pointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                mode='max')  # 保存验证集上准确率最高的模型
model.fit(x_train, y_train, batch_size=batch_size, epochs=30, verbose=2,validation_data=(x_val, y_val),shuffle=True, callbacks=[check_pointer])  # 模型训练
model.save('inception_4194.h5')
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# plot_model(model, to_file='inception_resnet.png')
