#inception_resnet 网络的测试数据
from keras.utils import np_utils
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import load_model
import csv
from scipy.io import loadmat
import scipy.io as sio
from sklearn.metrics import classification_report
####导入测试集----
#path3 = 'E:/毕设/代码/331_cnn/cnn_test_set.mat'
path3='E:/毕设/代码/407test/cnn/cnn_test_set.mat'
test = sio.loadmat(path3)
test2=test['cnn_test_set']

#信号总体正确识别率
p=np.zeros([1,26])
data=[]
for i in range(0,51,2):
    test2_20db = test2[:, :, i]
    x_test_20db = test2_20db[:, 0:3072]
    y_test_20db = test2_20db[:, -1]
    x_train, x_2, y_train, y_2 = model_selection.train_test_split(x_test_20db, y_test_20db, random_state=1,test_size=199)
    x_2=x_2.reshape(-1,1024,3)
    y_2 = np_utils.to_categorical(y_2, 6)
    model=load_model('inception_resnet_4201.h5')
    mark = ['.', 'o', '^', 's', 'D', '<', 'v', '*', 'd', 'h', '8']
    label = [0,1,2,3,4,5]
    m=i/2
    m=int(m);
    p[0,m] = model.evaluate(x_2, y_2, batch_size=256)[1]
    data.append(p)
    y_pred=model.predict(x_2,batch_size=256)
    sio.savemat('E:/毕设/代码/407test/cnn/in_resnet_4201.mat', {'in_resnet_4201': p})

#各信号正确识别率

p=np.zeros([6,26])
data=[]
for i in range(0,51,2):
    test2_20db = test2[:, :, i]
    for j in range(0, 6, 1):
        guo=test2_20db[200*j:200*(j+1),:]
        x_test_20db = guo[:, 0:3072]
        y_test_20db = guo[:, -1]
        x_train, x_2, y_train, y_2 = model_selection.train_test_split(x_test_20db, y_test_20db, random_state=1,test_size=100)
        x_2=x_2.reshape(-1,1024,3)
        y_2 = np_utils.to_categorical(y_2, 6)
        model=load_model('inception_resnet_4201.h5')
        mark = ['.', 'o', '^', 's', 'D', '<', 'v', '*', 'd', 'h', '8']
        label = [0,1,2,3,4,5]
        m=i/2;
        m=int(m)
        p[j,m] = model.evaluate(x_2, y_2, batch_size=256)[1]
        y_pred=model.predict(x_2,batch_size=256)
        sio.savemat('E:/毕设/代码/407test/cnn/in_resnet_4201ever.mat', {'in_resnet_4201ever': p})