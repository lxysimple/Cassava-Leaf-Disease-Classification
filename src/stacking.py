"""

配置环境：conda install scikit-learn

"""
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as ADA

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

# 自动生成6000个样本,每个样本有2个类别
x,y = make_classification(n_samples=6000)
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.5)

# 第一层模型
clfs = [GBDT(n_estimators=100),# Classifiers
        RF(n_estimators=100),
        ET(n_estimators=100),
        ADA(n_estimators=100)]

#GBDT分类器
clf_1 = clfs[0]
clf_1.fit(x_train,y_train)
# 取所有预测3000行的第二列,因为y_test就是第二列的结果
pred_1 = clf_1.predict_proba(x_test)[ :,1]
print("GBDT",roc_auc_score(y_test,pred_1))


#随机森林分类器
clf_2 = clfs[1]
clf_2.fit(x_train,y_train)
pred_2 = clf_2.predict_proba(x_test)[:,1]
print("随机森林",roc_auc_score(y_test,pred_2))

#ExtraTrees分类器
clf_3 = clfs[2]
clf_3.fit(x_train,y_train)
pred_3 = clf_3.predict_proba(x_test)[:,1]
print("ExtraTrees",roc_auc_score(y_test,pred_3))

#AdaBoost分类器
clf_4 = clfs[3]
clf_4.fit(x_train,y_train)
pred_4 = clf_4.predict_proba(x_test)[:,1]
print("AdaBoost",roc_auc_score(y_test,pred_4))

# x_train_stack:(3000,4),4个模型
x_train_stack = np.zeros((x_train.shape[0],len(clfs)))
# x_test_stack:(3000,4)
x_test_stack = np.zeros((x_test.shape[0],len(clfs)))


#6折stacking
n_folds = 6
# 我感觉skf是一个分类算法,把数据分成6份,每1份数据包含一组(x_train, y_train)
skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=1)
for i,clf in enumerate(clfs):
    # x_stack_test_n:(3000,6)
    x_stack_test_n = np.zeros((x_test.shape[0],n_folds))
    for j, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
        # train_index:(2500),train:5*500,test:1*500
        tr_x = x_train[train_index] # 获得2500个训练样本
        tr_y = y_train[train_index] # 获得2500个标签

        clf.fit(tr_x,tr_y)
        # 获得本次500个测试样本的预测数据,一共6折,所以刚好3000个预测结果
        # 将1个模型的6个子模型concat到1个向量中,4个模型是4个向量,结果拼成一个(3000,4)的一个矩阵
        x_train_stack[test_index,i] = clf.predict_proba(x_train[test_index])[:, 1]
        # 对当前折的模型进行3000个测试样本预测,结果的第二列存起来,1个模型要存6次
        x_stack_test_n[:, j] = clf.predict_proba(x_test)[:, 1]

    # 对1个模型的6次3000样本测试结果,进行一个取均值,结果存起来
    # 一共4个模型,x_test_stack:(3000,4)
    x_test_stack[:, i] = x_stack_test_n.mean(axis=1)


# 第二层模型LR,主要负责找出4个模型的优缺点,利用3个模型的优点去弥补1个模型的缺点
# 输入(n_samples, n_features),输出(n_samples,)是每个样本的分类标签
clf_second = LogisticRegression(solver="lbfgs")
# x_train_stack:(3000,4),y_train(3000)
clf_second.fit(x_train_stack,y_train)
pred = clf_second.predict_proba(x_test_stack)[:,1]
print("Stacking",roc_auc_score(y_test,pred))





