# 学习笔记大纲

[ML启程系列笔记](http://blog.csdn.net/zxlong7749/article/details/79099634)的同步笔记代码分享


# 一、开发环境搭建

## 1.1 Python环境搭建

#### 1.1.1 直接安装

#### 1.1.2 通过miniconda安装和基本命令

### 1.2 常用科学计算库

#### 1.2.1 Numpy安装和常用函函数的使用

#### 1.2.2 Matplotlib安装和常用函数的使用

#### 1.2.3 Sklearn 安装和基本使用

#### 1.2.4 Opencv-python 安装和基本图像处理

### 1.3 神经网络框架安装

#### 1.3.1 Tensorflow 安装和常用API参考

#### 1.3.2 Caffe 安装和入门Demo

### 1.4 开发工具

#### 1.4.1 Pycharm 安装和配置

#### 1.4.2 Jupyter notebook 安装和使用


# 二、机器学习基础概念

## 2.1 什么是机器学习

## 2.2 机器学习能做什么(生活中的机器学习)

## 2.3 机器学习怎么做(一般流程)

## 2.4 机器学习与数据挖掘的区别

## 2.5 机器学习相关的术语

## 2.6 机器学习算法的分类及选择算法的一般标准

# 三、机器学习基本算法

## 有监督学习算法

## 分类算法

### 3.1 knn算法

- K-nn的基本思想

### 3.2 决策树算法

- 决策树的基本思想
- 决策树的构造
    - ID3算法
    - C4.5算法
    - CART算法
- 决策树的绘制

### 3.3 朴素贝叶斯算法

- 贝叶斯准则
- 词集模型
    - 不考虑词汇出现的次数,所有词汇具有同等权重
- 词袋模型
    - 考虑词汇的出现次数,出现次数多的词汇具有较大的权重

### 3.4 Logistic回归
    设回归系数为W
    则输入为 z  = w0*x0 + W1*x1+ ...+wn*xn
    其中x 为特征取值
    输出为: Sigmoid(z) = 1/(1 + E**(-z))

### 3.5 优化算法

#### 3.5.1 梯度下降

#### 3.5.2 随机梯度下降

### 3.6 SVM算法

#### 3.6.1 基于SMO(序列最小化)的SVM实现

#### 3.6.2 基于核函数的SVM扩展

#### 3.6.3 LIBSVM库

## 分类器的性能度量

#### 3.7 混淆矩阵

#### 3.8 错误率

#### 3.9 正确率

#### 3.10 召回率

#### 3.11 ROC曲线

## 非均衡分类问题

### 3.12 代价敏感学习

### 3.13 数据抽样

### 3.14 欠抽样与过抽样

## 回归算法

### 3.15 线性回归

### 3.16 局部加权回归

### 3.17 基于缩减法的回归

#### 3.17.1 岭回归

#### 3.17.2 逐步线性回归

### 3.18 偏差与方差的折中

### 3.19 树回归

## 无监督学习算法

### 3.20 聚类算法

#### 3.20.1 K-mean算法

### 3.21 密度估计算法


# 四、神经网络算法

## 4.1 感知机

## 4.2 单隐层前馈网络

## 4.3 多隐层前馈网络

## 4.4 深度神经网络(DNN)

## 4.5 卷积神经网络(CNN)

### 4.5.1 LeNet-5

### 4.5.2 AlxNet

### 4.5.3 VGGNet-11 - VGGNet-19

### 4.5.4 InspetionNet

### 4.5.5 RestNet

### 4.5.6 SeNet

## 4.6 循环神经网络(RNN)

### 4.6.1长短时记忆网络(LSTM)

# 五、集成学习/增强学习/迁移学习

## 5.1 集成学习

### 5.1.1 Bagging方法(自举汇聚法)

#### 5.1.1.1 基于数据随机重抽样(放回取样)

#### 5.1.1.1Random forest

### 5.1.2 Boosting方法

#### 5.1.2.1 AdaBoost(自适应Boosting)

### 5.1.3 Q-Learning

### 5.1.4 DQN 及其变体

### 5.1.5 R-CNN及其迭代版本

- Fast RCNN
- Fater RCNN
- Mask RCNN
- Yolo V1

# 六、大数据与机器学习

## 6.1 Hadoop

## 6.2 Spark



