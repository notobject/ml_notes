import numpy as np
from pylab import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.vq import *
import mahotas as mh
from sklearn.decomposition import PCA
import os
import shutil
from skimage import transform
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)

src_path = "./data/red_boll/"
dst_path = "./data/dst_boll/"
IMAGE_SIZE = (32, 32)
K_NUM = 10


# 根据肘部法则,确定分类数
def select_k(features):
    K = range(1, K_NUM)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        meandistortions.append(sum(np.min(
            cdist(features, kmeans.cluster_centers_,
                  'euclidean'), axis=1)) / features.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel(u'平均畸变程度', fontproperties=font)
    plt.title(u'用肘部法则来确定最佳的K值', fontproperties=font)
    plt.show()
    key = input()
    return key


def show_datas(features):
    # 画出数据点聚类图
    centroids, variance = kmeans(features, K_NUM)
    code, distance = vq(features, centroids)
    figure()
    ndx = where(code == 0)[0]
    plot(features[ndx, 0], features[ndx, 1], '*')
    ndx = where(code == 1)[0]
    plot(features[ndx, 0], features[ndx, 1], 'r.')
    plot(centroids[:, 0], centroids[:, 1], 'go')

    title(u'数据点聚类', fontproperties=font)
    axis('off')
    show()


def get_data(root_path):
    files = os.listdir(root_path)
    features = None
    for index, file_name in enumerate(files):
        try:
            # 读取文件 并缩放到 0-1区间
            image = mh.imread(root_path + file_name)
            image = transform.resize(image, IMAGE_SIZE, mode="reflect")
            original_img = np.array(image, dtype=np.float64) / 255
            # 获取图片宽高,通道数量
            width, height, deepth = tuple(original_img.shape)
            # 将图片转化为二维矩阵[像数点数,通道数]
            image_flattend = np.reshape(original_img, (width * height, deepth))
            pcaData = PCA(n_components=1).fit_transform(image_flattend).reshape(1, -1)
            # 取第三通道将图片转换为向量
            if features is None:
                features = array([pcaData[0]])
            else:
                features = np.vstack((features, [pcaData[0]]))
        except OSError:
            os.remove(root_path + file_name)
            files.remove(index)
    print(features.shape)
    return files, features


def k_means_classify(features):
    print("正在聚类")
    estimator = KMeans(n_clusters=K_NUM)
    # features 只能是2d向量[batch_size,feature_size]
    estimator.fit(features)
    # 为原始图片的每个像素进行类的分配(聚类)
    cluster_assignments = estimator.predict(features)
    return cluster_assignments


if __name__ == "__main__":
    files, features = get_data(src_path)
    cluster_assignments = k_means_classify(features)
    length = len(files)

    # 为每个类别依次创建目录
    dir = dst_path + "%d/"
    for i in range(K_NUM):
        if not os._exists("./data/%d/" % (i)):
            os.mkdir("./data/%d/" % (i))

    # 将文件移动到目标文件夹
    for i, n in enumerate(cluster_assignments):
        shutil.move(src_path + files[i], dir % n + files[i])
        print("移动文件,还剩余:%d" % (length - i))
    print(cluster_assignments.shape)
