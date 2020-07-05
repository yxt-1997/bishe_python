#bp不运用kfold验证法
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import dump
from sklearn import model_selection
from joblib import load
from sklearn.metrics import classification_report
import scipy.io as sio
from sklearn import preprocessing
path2='E:/毕设/代码/407test/feature/train_pinjie2_150.mat'
feature=sio.loadmat(path2)
feature=feature['train_pinjie2']
a=0
b=0
# get training data
m,n=np.split(feature,(12000,),axis=0)
x ,y=np.split(n,(6,),axis=1)
#x=np.log(x)对数归一化
# scaler = StandardScaler()  # 标准化转换
# scaler.fit(x)  # 训练标准化对象
# x = scaler.transform(x)
minmaxscaler = preprocessing.MinMaxScaler().fit(x)#min-max标准化
minmaxscaler.transform(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=1000)
clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(6,), random_state=1)
clf.fit(x_train, y_train.ravel())
print('训练集准确率：', clf.score(x_train, y_train))
y_hat = clf.predict(x_train)
def show_accuracy(y_hat, y_train, str):
 pass
show_accuracy(y_hat, y_train, '训练集')
print('测试集准确率', clf.score(x_test, y_test))
y_hat = clf.predict(x_test)
show_accuracy(y_hat, y_test, '测试集')
dump(clf, "bp_buzhe_log.m")
#测试数据
path3='E:/毕设/代码/407test/feature/test_set.mat'
test_data=sio.loadmat(path3)
test_data=test_data['test_set']
p=np.zeros(26)
clf = load("bp_buzhe_log.m")
for i in range(0,51,2):
   x=test_data[:,:,i]
   x, y = np.split(x, (6,), axis=1)
   minmaxscaler.transform(x)
   #x=np.log(x)
   #x = scaler.transform(x)  # 转换数据集
   X, x_test, y, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=1000)
   y_pred = clf.predict(x_test)
   test_score = clf.score(x_test, y_test)
   test_score2=clf.score(X,y)
   print(classification_report(y_test, y_pred))
   m = i / 2
   m = int(m);
   p[m]=test_score
sio.savemat('E:/毕设/代码/407test/bp_svm/pre/bp_min_all.mat', {'bp_min_all': p})
##各种测试数据结果
test_data=sio.loadmat(path3)
test_data=test_data['test_set']
p=np.zeros(shape=(6,26))
clf=load("bp_buzhe_log.m")
for i in range(0,51,2):
   m=test_data[:,:,i];
   for j in range(0,6,1):
      x=m[200*j:200*(j+1),:]
      x,y=np.split(x,(6,), axis=1)
      #x = np.log(x)
      minmaxscaler.transform(x)
      #x = scaler.transform(x)  # 转换数据集
      x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=199)
      y_pred=clf.predict(x_test)
      q=i/2
      q=int(q)
      p[j,q]=clf.score(x_test,y_test)
      print(classification_report(y_test,y_pred))
sio.savemat('E:/毕设/代码/407test/bp_svm/pre/bp_min_ever.mat.mat', {'bp_min_ever': p})