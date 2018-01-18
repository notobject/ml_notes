# encoding=utf-8
# Created by Mr.Long on 2017/12/13 0013.
# 这是文件的概括
import numpy as np
import matplotlib.pyplot as plt
#我们计算K值从1到10对应的平均畸变程度：
from sklearn.cluster import KMeans
#用scipy求解距离
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)

# 随机生成一个实数，范围在（0.5,1.5）之间
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
# hstack拼接操作
X = np.hstack((cluster1, cluster2)).T

def main1():

    plt.figure()
    plt.axis([0, 5, 0, 5])
    plt.grid(True)

    plt.plot(X[:, 0], X[:, 1], 'k.')
    plt.show()
    plt.waitforbuttonpress()
    pass

def main2():

    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(
            cdist(X, kmeans.cluster_centers_,
                  'euclidean'), axis=1)) / X.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel(u'平均畸变程度', fontproperties=font)
    plt.title(u'用肘部法则来确定最佳的K值', fontproperties=font)
    plt.show()
    plt.waitforbuttonpress()

def main3():
    x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
    x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    plt.figure()
    plt.axis([0, 10, 0, 10])
    plt.grid(True)
    plt.plot(X[:, 0], X[:, 1], 'k.')
    plt.show()

def main4():
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist

    x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
    x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(
            X, kmeans.cluster_centers_, "euclidean"), axis=1)) / X.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel(u'平均畸变程度', fontproperties=font)
    plt.title(u'用肘部法则来确定最佳的K值', fontproperties=font)
    plt.show()

def main5():
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn import metrics

    plt.figure(figsize=(8, 10))
    plt.subplot(3, 2, 1)
    x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
    x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title(u'样本', fontproperties=font)
    plt.scatter(x1, x2)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
    tests = [2, 3, 4, 5, 8]
    subplot_counter = 1
    for t in tests:
        subplot_counter += 1
        plt.subplot(3, 2, subplot_counter)
        kmeans_model = KMeans(n_clusters=t).fit(X)
        #     print kmeans_model.labels_:每个点对应的标签值
        for i, l in enumerate(kmeans_model.labels_):
            plt.plot(x1[i], x2[i], color=colors[l],
                     marker=markers[l], ls='None')
            plt.xlim([0, 10])
            plt.ylim([0, 10])
            plt.title(u'K = %s, 轮廓系数 = %.03f' %
                      (t, metrics.silhouette_score
                      (X, kmeans_model.labels_, metric='euclidean'))
                      , fontproperties=font)
    plt.show()

def main6():
    import numpy as np
    import mahotas as mh

    original_img = np.array(mh.imread('test.jpg'), dtype=np.float64) / 255
    original_dimensions = tuple(original_img.shape)
    width, height, depth = tuple(original_img.shape)
    image_flattend = np.reshape(original_img, (width * height, depth))

    print(image_flattend.shape)
    print(image_flattend)

if __name__ == "__main__":
    main6()
