# encoding=utf-8
# Created by Mr.Long on 2017/12/13 0013.
# 这是文件的概括

import cv2
import numpy as np
import time

cap = cv2.VideoCapture(1)

# 全局变量
window_name = "Demo"
H, S, V = (2, 80, 46)
HD, SD, VD = (10, 255, 255)


def main():
    cv2.namedWindow(window_name)

    while True:
        _, frame = cap.read()
        # 将图片转换为灰度图,在这里不需要
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 将图片从 BGR 空间转换到 HSV 空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义在HSV空间中颜色的范围
        lower_blue = np.array([H, S, V])
        upper_blue = np.array([HD, SD, VD])

        # 根据以上定义的阈值得到感兴趣的的部分
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 滤波降噪-高斯滤波
        blur = cv2.GaussianBlur(mask, (1, 1), 15, 1)

        # 封闭填充
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

        # 形态学腐蚀与膨胀
        closed = cv2.erode(closed, None, iterations=1)
        closed = cv2.dilate(closed, None, iterations=1)

        # Canny 边缘检测
        canny = cv2.Canny(closed, 10, 150)
        canny = np.uint8(np.absolute(canny))

        # 选择轮廓
        (img, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        index = 0
        offset = 10
        for i in range(len(sorted_cnts)):
            try:
                rect = cv2.minAreaRect(sorted_cnts[0])
                box = np.int0(cv2.boxPoints(rect))
                Xs = [i[0] for i in box]
                Ys = [i[1] for i in box]
                x1 = min(Xs) - offset
                x2 = max(Xs) + offset
                y1 = min(Ys) - offset
                y2 = max(Ys) + offset
                hight = y2 - y1
                width = x2 - x1
                box[0] = [x1, y1]
                box[1] = [x2, y1]
                box[2] = [x2, y2]
                box[3] = [x1, y2]
                if width <= 11 or hight <= 11:
                    continue
                if width > 80 or hight > 80:
                    continue
                if 0.99 < hight / width < 1.01:
                    # //print(y1, y1 + hight, x1, x1 + width, width, hight)
                    print(box)
                    cropImg = frame[y1:y1 + hight, x1:x1 + width]
                    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
                    break
            except:
                continue
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
