# 学姐模型原代码
from keras.models import Model
from keras.layers import (Input,Activation,Dense,Flatten,Reshape,Concatenate)
from keras.layers.convolutional import (Conv2D,MaxPooling2D,AveragePooling2D)
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import initializers
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES']="2"#选择GPU Device 2
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 参数
channel=2
class_num =6          #干扰类别数
learning_rate = 0.01    #梯度更新学习率
epoch = 2000            #样本遍历的次数
batch_size = 256        #更新一次梯度所用样本数
momentum = 0.85           #Batch norm指数加权平均的系数
data_len=1024

filepath='inception_resnet.h5'

initializer=initializers.he_normal(seed=None)#定义权重初始化方式
# 训练或者测试
# is_train=False
is_train = True
# is_test=True
is_test = False
# 加载验证数据集
path2 = 'valid.mat'
validation_x = sio.loadmat(path2)
validation_x = np.array(
    [validation_x['ST'], validation_x['MT'], validation_x['C'], validation_x['PB'],
     validation_x['FM'], validation_x['Pulse']]).reshape(-1, data_len, channel)
validation_y = []
for i in range(class_num):
    yy = i * np.ones(11 * 300)
    validation_y.append(yy)
validation_y = np.array(validation_y).ravel()
validation_y = np_utils.to_categorical(validation_y, num_classes=class_num)
if is_train:
    # 加载训练数据集
    path1 = 'train.mat'
    x_train = sio.loadmat(path1)
    x_train = np.array(
        [x_train['ST'], x_train['MT'], x_train['C'], x_train['PB'], x_train['FM'],x_train['Pulse']]).reshape(
        -1, data_len, channel)
    y_train2 = []
    for i in range(class_num):
        yy = i * np.ones(31 * 300)
        y_train2.append(yy)
    y_train2 = np.array(y_train2).ravel()
    print(y_train2.shape)
    y_train = np_utils.to_categorical(y_train2, num_classes=class_num)

    #构建Inception A和Inception ResNet A模块
    def inception_resnet_function(x_input, filter_midille,filter):
        # Inception A-------------------------------------------------------------
        # 1*1
        conv_1x1_a_A = Conv2D(filters=filter, kernel_size=(1, 1), strides=(2, 1), padding="same",
                        kernel_initializer=initializer)(x_input)
        f_a = Activation('relu')(conv_1x1_a_A)
        # 1*1+3*1
        conv_1x1_b_A = Conv2D(filters=filter_midille, kernel_size=(1, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(x_input)
        f_ba = Activation('relu')(conv_1x1_b_A)
        conv_3x1_b_A = Conv2D(filters=filter, kernel_size=(3, 1), strides=(2, 1), padding="same",
                         kernel_initializer=initializer)(f_ba)
        f_b = Activation('relu')(conv_3x1_b_A)
        # MaxPool+1*1--------------------------------------------------
        max_A = MaxPooling2D((3, 1), strides=(2, 1), padding="same")(x_input)
        conv_1x1_c_A = Conv2D(filters=filter, kernel_size=(1, 1), strides=(1, 1), padding="same",
                        kernel_initializer=initializer)(max_A)
        f_d = Activation('relu')(conv_1x1_c_A)
        # Concatenate----------------------------------------------------------
        inceptionA = Concatenate(axis=3)([f_a, f_b, f_d])
        x1= BatchNormalization(axis=-1, momentum=momentum)(inceptionA)
        # Inception ResNet A---------------------------------------------------
        # 1*1
        conv_1x1_a_B = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same",
                        kernel_initializer=initializer)(x1)
        f_a_B = Activation('relu')(conv_1x1_a_B)
        # 1*1+3*1
        conv_1x1_b_B = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(x1)
        f_b1_B = Activation('relu')(conv_1x1_b_B)
        conv_3x1_b_B = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(f_b1_B)
        f_b_B = Activation('relu')(conv_3x1_b_B)
        # 1*1+3*1+3*1
        conv_1x1_c_B = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(x1)
        f_c1_B = Activation('relu')(conv_1x1_c_B)
        conv_3x1_c1_B = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(f_c1_B)
        f_c2_B = Activation('relu')(conv_3x1_c1_B)
        conv_3x1_c2_B = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding="same",
                         kernel_initializer=initializer)(f_c2_B)
        f_c_B = Activation('relu')(conv_3x1_c2_B)
        x2 = Concatenate(axis=3)([f_a_B,f_b_B, f_c_B ])
        x_1x1=Conv2D(filters=filter*3, kernel_size=(1, 1), strides=(1, 1), padding="same",
               kernel_initializer=initializer)(x2)
        x3=layers.add([x1,x_1x1])
        return x3

    # 构建模型
    input = Input(shape=(1024, 2))
    input_reshape = Reshape((1024,1, 2 ))(input)
    x = Conv2D(filters=16, kernel_size=(5, 1), strides=(2, 1), padding="same", kernel_initializer=initializer)(input_reshape)
    x = Activation('relu')(x)
    print(x.shape)
    x = MaxPooling2D((3, 1), strides=(2, 1), padding="same")(x)
    print('conv1', x.shape)
    x = Conv2D(filters=16, kernel_size=(3, 1), strides=(2, 1), padding="same", kernel_initializer=initializer)(x)
    conv_2 = Activation('relu')(x)
    print('conv2', conv_2.shape)
    conv3_a = MaxPooling2D((3, 1), strides=(2, 1), padding="same")(conv_2)
    conv3_b1 = Conv2D(filters=16, kernel_size=(3, 1), strides=(2, 1), padding="same", kernel_initializer=initializer)(conv_2)
    conv3_b= Activation('relu')(conv3_b1)
    conv3_all_c= Concatenate(axis=3)([conv3_a, conv3_b])
    conv3_all=BatchNormalization(axis=-1, momentum=momentum)(conv3_all_c)
    print('Inception concatenate',conv3_all.shape)
    print('Inception-ResNet1 begin')
    x = inception_resnet_function(conv3_all, 8,16)
    x = Activation('relu')(x)
    x=BatchNormalization(axis=-1, momentum=momentum)(x)
    print('Inception-ResNet1 end', x.shape)
    print('Inception-ResNet2 begin')
    x = inception_resnet_function(x, 8,16)
    x = Activation('relu')(x)
    x=BatchNormalization(axis=-1, momentum=momentum)(x)
    print('Inception-ResNet2 end', x.shape)
    print('Inception-ResNet3 begin')
    x = inception_resnet_function(x,8,16)
    x = Activation('relu')(x)
    x=BatchNormalization(axis=-1, momentum=momentum)(x)
    print('Inception-ResNet3 end', x.shape)
    x = AveragePooling2D((4, 1), strides=(4, 1), padding="same")(x)
    flatten = Flatten(name='flatten')(x)
    print('Flatten', flatten.shape)
    predict = Dense(class_num, activation='softmax', name='fc', kernel_initializer='he_normal')(flatten)
    model = Model(inputs=input, outputs=predict)
    adam = optimizers.adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    check_pointer=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')#保存验证集上准确率最高的模型
    model.fit(x_train,y_train,batch_size=batch_size,epochs=epoch,verbose=2,validation_data=(validation_x,validation_y),shuffle=True,callbacks=[check_pointer])#模型训练
if is_test:
    #加载测试数据集
    path3 = u'test_set.mat'
    test = sio.loadmat(path3)
    x_test=np.array([test ['STJ'],test ['MTJ'],test ['WNBJ'], test['NBNJ'],
                     test ['LFM'],test ['NFM'],test ['SFM'],
                     test['AM'],test ['BFSK'],test['BPSK'], test['QPSK']]).reshape(-1,4096,3)
    test_y =[]
    for i in range(class_num ):
        yy=i*np.ones(16*500)
        test_y.append(yy)
    test_y =np.array(test_y).ravel()
    y_test=np_utils.to_categorical(test_y,num_classes=class_num )
    model=load_model('model.h5')#加载模型
    mark = ['.', 'o', '^', 's', 'D', '<', 'v', '*', 'd', 'h', '8']
    label = [u'STJ', u'MTJ', u'WBNJ', u'NBNJ',u'LFM', u'NFM', u'SFM', u'AM',u'BFSK', u'BPSK', u'QPSK']
    xtest =x_test.reshape(class_num , 16, 500, 4096, 3)
    ytest =y_test.reshape(class_num , 16, 500, class_num )
    for i in np.arange(class_num ):
        data = []
        for j in np.arange(0, 16):
            p = model.evaluate(xtest[i][j], ytest[i][j], batch_size=256)[1]
            data.append(p)
        plt.plot(np.arange(-10, 21, 2), data, label=label[i],marker=mark[i],lw=2)
    plt.xticks(np.arange(-10, 21, 2))
    plt.xlabel('JNR(dB)', fontsize=13)
    plt.ylabel(u'识别概率', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.axis([-10, 20, 0, 1.05])
    plt.legend(loc='best', fancybox=True, framealpha=0.8, fontsize=9)
    plt.show()
