# encoding=utf-8
# Created by Mr.Long on 2017/9/5 0005.
# Matplotlib的基本使用




def main1():
    import numpy as np
    import matplotlib.pyplot as plt
    # line
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    c, s = np.cos(x), np.sin(x)
    plt.figure(1)
    plt.plot(x, c, color="blue", linewidth=1.0, linestyle="-", label="Cos", alpha=0.5)
    plt.plot(x, s, "r--", label="Sin")
    plt.title("Cos & Sin")
    plt.show()
    pass

def main2():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(1)  # 创建图表1
    plt.figure(2)  # 创建图表2
    ax1 = plt.subplot(211)  # 在图表2中创建子图1
    ax2 = plt.subplot(212)  # 在图表2中创建子图2
    x = np.linspace(0, 3, 100)
    for i in range(5):
        plt.figure(1)  # 选择图表1
        plt.plot(x, np.exp(i * x / 3))
        plt.sca(ax1)  # 选择图表2的子图1
    plt.plot(x, np.sin(i * x))
    plt.sca(ax2)  # 选择图表2的子图2
    plt.plot(x, np.cos(i * x))
    plt.show()

if __name__ == "__main__":
    main2()
