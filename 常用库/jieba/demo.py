# encoding=utf-8
# Created by Mr.Long on 2018/1/18 0018.
# 这是文件的概括


def main():
    import jieba

    seg_list = jieba.cut(u"我来到北京清华大学", cut_all=True)
    print("Full Mode:", "/".join(seg_list))  # 全模式

    seg_list = jieba.cut(u"我来到北京清华大学", cut_all=False)
    print("Default Mode:", "/".join(seg_list))  # 精确模式

    seg_list = jieba.cut(u"他来到了网易,杭研大厦")  # 默认是精确模式
    print("/".join([word for word in seg_list if len(word) > 1]))

    seg_list = jieba.cut_for_search(u"小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    print("/".join(seg_list))

    pass


if __name__ == "__main__":
    main()
