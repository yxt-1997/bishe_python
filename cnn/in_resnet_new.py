###inception-resnet结合网络--------
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
import scipy.io as sio
##基本参数设置
from keras.utils.vis_utils import plot_model
import  os
epoch=100
momentum=0.85
nb_features = 1024
nb_class = 6
batch_size=256
learning_rate=0.01
class_num=6
initializer=initializers.he_normal(seed=None)#定义权重初始化方式

#导入训练模型数据---
# path2='E:/毕设/代码/cnn/F.mat'
# test = sio.loadmat(path2)
# test2=test['F']
# x_all1=test2[:,0:3072]
# y_all1=test2[:,-1]
#path2='E:/毕设/代码/331_cnn/cnn_train_pinjie.mat'
path2='E:/毕设/代码/407test/cnn/cnn_train_pinjie.mat'
test = sio.loadmat(path2)
test2=test['cnn_train_pinjie']
#由于生成的样本JNR为-20开始，所以前12000个数据为JNR-10db以下的数据，不参与训练过程
x_all1=test2[12000:, 0:3072]
y_all1=test2[12000:,-1]
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_all1, y_all1, random_state=1, test_size=24000)
print("x_train shape",x_train.shape)
print("y_train.shape",y_train.shape)
x_train=x_train.reshape(-1,1024,3)
print("x_train shape",x_train.shape)
x_val=x_val.reshape(-1,1024,3)
y_train = np_utils.to_categorical(y_train, nb_class)#转换成独热码
y_val= np_utils.to_categorical(y_val, nb_class)
filepath='inception_resnet.h5'
    #构建Inception A和Inception ResNet A模块
def inception_resnet_function(x_input, filter_midille,fil):
        # Inception A-------------------------------------------------------------
        # 1*3
        print('before function',x_input.shape)
        conv_1x1_a_A = Conv2D(filters=fil, kernel_size=(3, 1), strides=(2, 1), padding="same",
                        kernel_initializer=initializer)(x_input)
        f_a = Activation('relu')(conv_1x1_a_A)
        print('reduction 分支1',f_a.shape)
        # 1*1+3*1+3*1
        conv_1x1_b_A = Conv2D(filters=filter_midille, kernel_size=(1, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(x_input)
        f_ba = Activation('relu')(conv_1x1_b_A)
        conv_3x1_b_A = Conv2D(filters=filter_midille, kernel_size=(3, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(f_ba)
        conv_3x1_b_A = Conv2D(filters=fil, kernel_size=(3, 1), strides=(2, 1), padding="same",
                              kernel_initializer=initializer)(conv_3x1_b_A)
        f_b = Activation('relu')(conv_3x1_b_A)
        print('reduction 分支2', f_b.shape)
        # MaxPool+1*1-------------------------------------------------
        max_A = MaxPooling2D((3, 1), strides=(2, 1), padding="same")(x_input)
        conv_1x1_c_A = Conv2D(filters=fil, kernel_size=(1, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(max_A)
        f_d = Activation('relu')(conv_1x1_c_A)
        print('reduction 分支3', f_d.shape)
        # Concatenate----------------------------------------------------------
        inceptionA = Concatenate(axis=3)([f_a, f_b, f_d])
        print('after reduction:',inceptionA.shape)
        x1= BatchNormalization(axis=-1, momentum=momentum)(inceptionA)
        # Inception ResNet A---------------------------------------------------
        # 1*1
        conv_1x1_a_B = Conv2D(filters=fil, kernel_size=(1, 1), strides=(1, 1), padding="same",
                        kernel_initializer=initializer)(x1)
        f_a_B = Activation('relu')(conv_1x1_a_B)
        print('resnet 分支1', f_a_B.shape)
        # 1*1+3*1
        conv_1x1_b_B = Conv2D(filters=fil, kernel_size=(1, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(x1)
        f_b1_B = Activation('relu')(conv_1x1_b_B)
        conv_3x1_b_B = Conv2D(filters=fil, kernel_size=(3, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(f_b1_B)
        f_b_B = Activation('relu')(conv_3x1_b_B)
        print('redsnet 分支2', f_b_B.shape)
        # 1*1+3*1+3*1
        conv_1x1_c_B = Conv2D(filters=fil, kernel_size=(1, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(x1)
        f_c1_B = Activation('relu')(conv_1x1_c_B)
        conv_3x1_c1_B = Conv2D(filters=fil, kernel_size=(3, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(f_c1_B)
        f_c2_B = Activation('relu')(conv_3x1_c1_B)
        conv_3x1_c2_B = Conv2D(filters=fil, kernel_size=(3, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(f_c2_B)
        f_c_B = Activation('relu')(conv_3x1_c2_B)
        print('resnet 分支3', f_c_B.shape)
        x2 = Concatenate(axis=3)([f_a_B,f_b_B, f_c_B ])
        print('after concatanate', x2.shape)
        x_1x1=Conv2D(filters=3*fil, kernel_size=(1, 1), strides=(1, 1), padding="same",
               kernel_initializer=initializer)(x2)
        print('after cov', x_1x1.shape)
        x3=layers.add([x1,x_1x1])
        return x3

# 构建模型
input = Input(shape=(1024, 3))
input_reshape = Reshape((1024,1, 3 ))(input)
x = Conv2D(filters=16, kernel_size=(5, 1), strides=(2, 1), padding="same", kernel_initializer=initializer)(
    input_reshape)
x = Activation('relu')(x)
print(x.shape)
x = MaxPooling2D((3, 1), strides=(2, 1), padding="same")(x)
print('conv1', x.shape)
x = Conv2D(filters=16, kernel_size=(3, 1), strides=(2, 1), padding="same", kernel_initializer=initializer)(x)
conv_2 = Activation('relu')(x)
print('conv2', conv_2.shape)
conv3_a = MaxPooling2D((3, 1), strides=(2, 1), padding="same")(conv_2)
print('conv3_a', conv3_a.shape)
conv3_b1 = Conv2D(filters=16, kernel_size=(3, 1), strides=(2, 1), padding="same", kernel_initializer=initializer)(
    conv_2)
print('conv3_b1', conv3_b1.shape)
conv3_b = Activation('relu')(conv3_b1)
print('conv3_b', conv3_b.shape)
conv3_all_c = Concatenate(axis=3)([conv3_a, conv3_b])
print('conv3_all_c', conv3_all_c.shape)
conv3_all = BatchNormalization(axis=-1, momentum=momentum)(conv3_all_c)
print('Inception concatenate', conv3_all.shape)
print('Inception-ResNet1 begin')
x = inception_resnet_function(conv3_all, 4,8)
x = Activation('relu')(x)
x = BatchNormalization(axis=-1, momentum=momentum)(x)
print('Inception-ResNet1 end', x.shape)
print('Inception-ResNet2 begin')
x = inception_resnet_function(x, 8, 16)
x = Activation('relu')(x)
x = BatchNormalization(axis=-1, momentum=momentum)(x)
print('Inception-ResNet2 end', x.shape)
print('Inception-ResNet3 begin')
x = inception_resnet_function(x, 16, 32)
x = Activation('relu')(x)
x = BatchNormalization(axis=-1, momentum=momentum)(x)
print('Inception-ResNet3 end', x.shape)
x = AveragePooling2D((8, 1), strides=(8, 1), padding="same")(x)
print("after average",x.shape)
flatten = Flatten(name='flatten')(x)
print('Flatten', flatten.shape)
predict = Dense(class_num, activation='softmax', name='fc', kernel_initializer='he_normal')(flatten)
print('after dense', predict.shape)
model = Model(inputs=input, outputs=predict)
adam = optimizers.adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
check_pointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                mode='max')  # 保存验证集上准确率最高的模型
model.fit(x_train, y_train, batch_size=batch_size, epochs=30, verbose=2,validation_data=(x_val, y_val),shuffle=True, callbacks=[check_pointer])  # 模型训练
model.save('inception_resnet_4201.h5')
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model, to_file='inception_resnet.png')