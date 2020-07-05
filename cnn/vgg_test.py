
from keras.utils import np_utils
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import load_model
import scipy.io as sio
from sklearn.metrics import classification_report
####导入测试集----
#path3 = 'E:/毕设/代码/331_cnn/cnn_test_set.mat'
path3='E:/毕设/代码/407test/cnn/cnn_test_set.mat'
test = sio.loadmat(path3)
test2=test['cnn_test_set']


#各种信号在不同JNR下单独正确识别率------------
p=np.zeros([6,26])
data=[]
for i in range(0,51,2):
    test2_20db = test2[:, :, i]
    #test2_20db=test2_20db[0:200,:]
    for j in range(0,6,1):
        guo=test2_20db[200*j:200*(j+1),:]
        x_test_20db = guo[:, 0:3072]
        y_test_20db = guo[:, -1]
        x_train, x_2, y_train, y_2 = model_selection.train_test_split(x_test_20db, y_test_20db, random_state=1,test_size=100)
        #x_2=x_2.reshape(-1,1024,3)
        y_2 = np_utils.to_categorical(y_2, 6)
        model=load_model('vgg4181.h5')
        mark = ['.', 'o', '^', 's', 'D', '<', 'v', '*', 'd', 'h', '8']
        label = [0,1,2,3,4,5]
        m=i/2;
        m=int(m)
        X_test_r = np.zeros((len(x_2), 1024, 3))
        X_test_r[:, :, 0] = x_2[:, :1024]
        X_test_r[:, :, 1] = x_2[:, 1024:2048]
        X_test_r[:, :, 2] = x_2[:, 2048:]
        p[j,m] = model.evaluate(X_test_r, y_2, batch_size=256)[1]
        y_pred=model.predict(X_test_r,batch_size=256)
        sio.savemat('E:/毕设/代码/407test/cnn/vgg_418ever.mat', {'vgg_418ever': p})

#总体正确识别率
p=np.zeros([1,26])
for i in range(0,51,2):
    test2_20db = test2[:, :, i]
    #test2_20db=test2_20db[0:200,:]
    x_test_20db = test2_20db[:, 0:3072]
    y_test_20db = test2_20db[:, -1]
    x_train, x_2, y_train, y_2 = model_selection.train_test_split(x_test_20db, y_test_20db, random_state=1,test_size=1000)
    #x_2=x_2.reshape(-1,1024,3)
    y_2 = np_utils.to_categorical(y_2, 6)
    model=load_model('vgg4181.h5')
    mark = ['.', 'o', '^', 's', 'D', '<', 'v', '*', 'd', 'h', '8']
    label = [0,1,2,3,4,5]
    m=i/2;
    m=int(m)
    X_test_r = np.zeros((len(x_2), 1024, 3))
    X_test_r[:, :, 0] = x_2[:, :1024]
    X_test_r[:, :, 1] = x_2[:, 1024:2048]
    X_test_r[:, :, 2] = x_2[:, 2048:]
    p[0,m] = model.evaluate(X_test_r, y_2, batch_size=256)[1]
    y_pred=model.predict(X_test_r,batch_size=256)
    sio.savemat('E:/毕设/代码/407test/cnn/vgg_418all.mat', {'vgg_418all': p})