# coding=utf-8
import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("中华人民共和国主席令第四十二号", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("中华人民共和国主席令第四十二号")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("中华人民共和国主席令第四十二号")  # 搜索引擎模式
print(", ".join(seg_list))