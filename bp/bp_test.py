#测试数据总体结果---------
#path3='D:/matlab2019/bin/M文件/毕设/402/test_set.mat'
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from joblib import dump
from sklearn import svm
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.metrics import classification_report
import scipy.io as sio
from sklearn import preprocessing
path3='E:/毕设/代码/407test/feature/test_set.mat'
test_data=sio.loadmat(path3)
test_data=test_data['test_set']
p=np.zeros(26)
clf = load("bp_407.m")
for i in range(0,51,2):
   x=test_data[:,:,i]
   x, y = np.split(x, (6,), axis=1)
   # x=np.log(x)
   # x = scaler.transform(x)  # 转换数据集
   X, x_test, y, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=1000)
   y_pred = clf.predict(x_test)
   test_score = clf.score(x_test, y_test)
   test_score2=clf.score(X,y)
   print(classification_report(y_test, y_pred))
   m = i / 2
   m = int(m);
   p[m]=test_score
sio.savemat('E:/毕设/代码/407test/bp_svm/bp407test.mat', {'bp407': p})
##各种测试数据结果

test_data=sio.loadmat(path3)
test_data=test_data['test_set']
p=np.zeros(shape=(6,26))
clf=load("bp_407.m")
for i in range(0,51,2):
   m=test_data[:,:,i];
   for j in range(0,6,1):
      x=m[200*j:200*(j+1),:]
      x,y=np.split(x,(6,), axis=1)
      #x = scaler.transform(x)  # 转换数据集
      x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=199)
      y_pred=clf.predict(x_test)
      q=i/2
      q=int(q)
      p[j,q]=clf.score(x_test,y_test)
      print(classification_report(y_test,y_pred))
#
sio.savemat('E:/毕设/代码/407test/bp_svm/bp407_evertets.mat', {'bp407_ever': p})