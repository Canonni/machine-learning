from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


np.random.seed(1000)
file_path = 'airfoil_self_noise.dat'
if __name__ == '__main__':
    # 载入数据集
    df = pd.read_csv(file_path, sep='\t', header=None)

    # 提取自变量和因变量
    X = df.iloc[:, 0:5].values
    Y = df.iloc[:, 5].values

    # 遍历数据集
    ssx, ssy = StandardScaler(), StandardScaler()
    Xs = ssx.fit_transform(X)
    Ys = ssy.fit_transform(Y.reshape(-1, 1))

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(Xs, Ys.ravel(), test_size=300, random_state=1000)

    # 训练SVR---RBF函数
    svr = SVR(kernel='rbf', gamma=0.75, C=2.8, cache_size=500, epsilon=0.1)
    svr.fit(X_train, Y_train)

    # 输出R^2
    print('Training R^2 score: %.3f' % svr.score(X_train, Y_train))
    print('Test R^2 score: %.3f' % svr.score(X_test, Y_test))

    # 显示原始数据与预测数据图像
    plt.plot(ssy.inverse_transform(Ys), color='green', label='Original dataset')
    plt.plot(ssy.inverse_transform(svr.predict(Xs)), color='red', label='Predictions')
    plt.legend() # 显示图例
    plt.xlabel('Sample')
    plt.ylabel('Scaled sound pressure (dB)')
    plt.grid()
    plt.show()

    # 显示绝对误差图像
    # fig, ax = plt.subplots(figsize=(15, 4))

    # Y = np.squeeze(ssy.inverse_transform(Ys))
    # Yp = ssy.inverse_transform(svr.predict(Xs))

    # ax.plot(np.abs(Y - Yp))
    # ax.set_title('Absolute errors')
    # ax.set_xlabel('Sample')
    # ax.set_ylabel(r'$|Y-Yp|$')
    # ax.grid()
    # plt.show()