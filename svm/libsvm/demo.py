# encoding=utf-8
# Created by Mr.Long on 2017/11/21 0021.
# 这是文件的概括

import svmutil as su


def main():
    y, x = su.svm_read_problem("./libsvm-3.22/heart_scale")
    m = su.svm_train(y[:200], x[:200], "-s 0 -c 4")
    p_label, p_acc, p_val = su.svm_predict(y[200:], x[200:], m)
    # print(p_label)
    # print(p_acc)
    # print(p_val)
    pass


if __name__ == "__main__":
    main()
