
###k折交叉验证#######
####保存模型的有训练测试验证集的bp神经网络######
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from joblib import dump
import numpy
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.metrics import classification_report
import scipy.io as sio
from sklearn import preprocessing
path2='D:/matlab2019/bin/M文件/毕设/402/train_pinjie.mat'
feature=sio.loadmat(path2)
feature=feature['train_pinjie']
a=0
b=0
# get training data
m,n=np.split(feature,(12000,),axis=0)
x ,y=np.split(n,(6,),axis=1)
x=np.log(x)
scaler = StandardScaler()  # 标准化转换
scaler.fit(x)  # 训练标准化对象
# x = scaler.transform(x)
# minmaxscaler = preprocessing.MinMaxScaler().fit(x)
# minmaxscaler.transform(x)
X, x_test1, y, y_test1 = model_selection.train_test_split(x, y, random_state=1, test_size=2000)
# neural network classifier of structure (3,2)
kf = KFold(n_splits=10)  # 3-fold cross-validation
best_clf = None
best_score = 0
train_scores = []
test_scores = []
print("kfold-------")
for train_index, test_index in kf.split(X):
   # create neural network using MLPClassifer
   clf = MLPClassifier(solver='adam', alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   clf.fit(X_train, y_train)
   train_score = clf.score(X_train, y_train)
   train_scores.append(train_score)

   test_score = clf.score(X_test, y_test)
   test_scores.append(test_score)

    # compare score of the tree models and get the best one
   if test_score > best_score:
        best_score = test_score
        best_clf = clf

    # print(clf.n_outputs_)
error=train_scores
in_sample_error = [1 - score for score in train_scores]
test_set_error = [1 - score for score in test_scores]
print("训练集准确率 ")
#print(in_sample_error)
for score in train_scores:
   b=b+score
b=b/5
train_correct=b
print(train_correct)
# print(train_scores)

print("验证集准确率：")
for score in test_scores:
   a=a+score
val_correct=a/5
print(val_correct)
#print(test_scores)

# store the classifier
if best_clf != None:
   dump(best_clf, "bp_log1.m")
#测试数据---------
path3='D:/matlab2019/bin/M文件/毕设/402/test_set.mat'
test_data=sio.loadmat(path3)
test_data=test_data['test_set']
p=np.zeros(51)
for i in range(0,51):
   x=test_data[:,:,i]
   x, y = np.split(x, (6,), axis=1)
   x=np.log(x)
   x = scaler.transform(x)  # 转换数据集
   X, x_test, y, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=100)
   clf = load("bp_log1.m")
   y_pred = clf.predict(x_test)
   test_score = clf.score(x_test, y_test)
   test_score2=clf.score(X,y)
   print(classification_report(y_test, y_pred))
   p[i]=test_score
sio.savemat('D:/matlab2019/bin/M文件/毕设/402/bp_log1.mat', {'bp_log1': p})
# for i in range():
# X_test = x_test1
# y_test = y_test1
# clf = load("train_model.m")
# y_pred = clf.predict(X_test)
# #np.savetxt("label_pred.txt", np.array(y_pred)) #save predict result
# #print(y_pred)
# test_score = clf.score(X_test, y_test)
# test_error = 1 - test_score
# test_correct=test_score
# target_names = ['0', '1', '2', '3', '4', '5']
# print(classification_report(y_test, y_pred, target_names=target_names))
# classification_report(y_test, y_pred)
# print('test_score：%s' % test_score)
# print('test_error：%s' % test_error)