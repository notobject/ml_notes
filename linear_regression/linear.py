# encoding=utf-8
# Created by Mr.Long on 2017/12/17 0017.
# Tensorflow与sklearn 构建线性回归的比较
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from linear_regression.linear_regression_model import LinearRegressionModel as lrm

if __name__ == '__main__':
    x, y = make_regression(7000)
    plt.figure(1)
    plt.plot(y, x, ".")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    print(x_train.shape)
    print(y_train.shape)
    y_lrm_train = y_train.reshape(-1, 1)
    y_lrm_test = y_test.reshape(-1, 1)
    print(x_train.shape)
    print(y_lrm_train.shape)

    # tensorflow_svm LinearRegression
    linear = lrm(x.shape[1])
    linear.train(x_train, y_lrm_train, x_test, y_lrm_test)
    y_predict = linear.predict(x_test)
    print("Tensorflow R2: ", r2_score(np.ravel(y_predict), np.ravel(y_lrm_test)))
    plt.figure(2)
    plt.plot(np.array(y_predict).reshape(-1, 1), x_test, ".")

    # sklearn LinearRegression
    lr = LinearRegression()
    y_predict = lr.fit(x_train, y_train).predict(x_test)
    print(y_predict)
    print("Sklearn R2: ", r2_score(y_predict, y_test))  #

    plt.figure(3)
    plt.plot(np.array(y_predict).reshape(-1, 1), x_test, ".")
    plt.show()
