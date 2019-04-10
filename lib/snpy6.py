# coding: utf-8

# @Time    : 2019/04/02
# @Author  : WANG JINGE
# @Site    :
# @File    : snpy6.py
# @Software: PyCharm
"""
    此类用于文本分析
"""

import pandas as pd
import networkx as nx
import matplotlib
import collections
import numpy as np
import json
import jieba
import jieba.posseg as pseg
import math
import itertools
import time


class GraphJson:

    def __init__(self, data, class_column, description_column):
        """
        传入dataframe
        :param data: 数据pandas.dataframe, 有两列，第一列类别，第二列描述
        :param class_column: 定义的类别列名
        :param description_column: 定义的描述列名
        """
        self.data = data
        self.class_column = class_column
        self.description_column = description_column
        self._repeat_class_word = set()
        self._class_name = []
        self._uncleaned_word_list = []

    def _iter_segment(self, flags):
        """
        描述分词，用‘ ’分隔，去除描述分词中和类别冲突的词，并用其类别词连接作为第一个词。生成器版本
        :param flags: 指定保留的分词flags列表
        :return: 返回一个可以输出每行分词结果（字符串）的迭代器
        """
        try:
            flags[0]
        except Exception:
            print('请输入分词保留的flags列表')
            exit()

        # jieba.enable_parallel()

        for row in self.data.itertuples(index=False):
            seg_list = []
            description_seg = pseg.cut(row[1].replace('\r\n', ''))
            words_seg = [word for word, flag in description_seg if flag in flags and word != row[0]]
            seg_list.append(row[0])
            seg_list.extend(words_seg)
            yield seg_list

    @staticmethod
    def _iter_remove_stopwords(seg_list, stopwords_line):
        """
        读取停用词列表，去除输入分词字符串中的停用词。生成器版本
        :param seg_line: 分词结果字符串
        :param stopwords_line: 停用词列表
        :return: 返回一个可以输出每行无停用词结果（字符串）的迭代器
        """
        without_stopwords = [word for word in seg_list if word not in stopwords_line]
        yield without_stopwords

    def seg_and_rm_stopwords(self, seg_flags, stopwords_relative_pos):
        """
        一步执行分词和去除停用词
        :param seg_flags: 指定保留的分词flags列表
        :param stopwords_relative_pos: 停用词相对路径位置
        :return: 返回一个列表，每项为字符串，每个字符串为类别词加分词结果，用‘ ’连接
        """
        try:
            with open(stopwords_relative_pos, encoding='utf-8') as sp:
                stopwords = sp.read()
                stopwords_line = stopwords.split('\n')
        except Exception:
            print('请确保停用词库在正确目录下')

        with_class_data = []
        for seg in self._iter_segment(seg_flags):
            for seg_rm in self._iter_remove_stopwords(seg, stopwords_line):
                with_class_data.append(seg_rm)

        with open('record1.txt', 'a', encoding='utf-8') as s:
            for line in with_class_data:
                for word in line:
                    s.write(word)
                    s.write(' ')
                s.write('\n')

        return with_class_data

    def _class_clean(self, class_threshold):
        """
        去重以及筛选满足阈值的类别
        :param class_threshold: 类别频率阈值
        :return: 清洗后的类别列表
        """
        class_cleaned = {class_name: count for class_name, count in collections.Counter(self.data[self.class_column]).items() if count > int(class_threshold)}

        try:
            max_count = max(class_cleaned.values())
            min_count = min(class_cleaned.values())
        except Exception:
            print('类别数据为空')
            raise

        try:
            for class_name, count in class_cleaned.items():
                class_cleaned[class_name] = 100*(0.3+((float(count)-min_count)/float(max_count-min_count)))
        except Exception:
            print('类别数据标准化出错')
            raise

        print(class_cleaned)
        return class_cleaned

    def _word_clean(self, with_class_list, word_threshold):
        """
        统计分词词频并筛选满足阈值的类别。
        :param with_class_list: 描述分词和去停用词后得到的字符串列表
        :param word_threshold: 分词频率阈值
        :return: 分词及其词频字典
        """
        without_class_list = []
        for line in with_class_list:
            without_class_list.append(line[1:])
        self._uncleaned_word_list = list(itertools.chain(*without_class_list))

        word_pool = {word: count for word, count in collections.Counter(self._uncleaned_word_list).items() if count > int(word_threshold)}

        try:
            max_count = max(word_pool.values())
            min_count = min(word_pool.values())
        except Exception:
            print('分词数据为空')

        try:
            for word_name, count in word_pool.items():
                word_pool[word_name] = 30*(0.1+((float(count)-min_count)/float(max_count-min_count)))
        except Exception:
            print('分词数据标准化出错')

        print(word_pool)
        return word_pool

    def gen_total_dict(self, with_class_list, word_threshold, class_threshold):
        """
        生成所有出现词的词频字典，若分词中出现和别的类别相同名称，则用‘_'分隔每个字重命名
        :param with_class_list: 描述分词和去停用词后得到的字符串列表
        :param word_threshold: 分词频率阈值
        :param class_threshold: 类别频率阈值
        :return: 所有出现词的词频字典
        """
        class_dict = self._class_clean(class_threshold)
        word_dict = self._word_clean(with_class_list, word_threshold)

        # 获取类别和分词中的相同名称列表，对应分词中用‘_'分隔每个字进行重命名，避免冲突
        class_set = set(self.data[self.class_column])
        word_set = set(self._uncleaned_word_list)
        self._repeat_class_word = class_set & word_set
        for repeat_name in self._repeat_class_word:
            if repeat_name in word_dict:
                word_dict['_'.join(repeat_name)] = word_dict.pop(repeat_name)

        self._class_name = class_dict.keys()
        total_dict = {**word_dict, **class_dict}
        return total_dict

    def gen_class_word_pool(self, with_class_list, total_dict):
        """
        生成类与对应出现过的词列表的字典
        :param with_class_list: 描述分词和去停用词后得到的字符串列表
        :param total_dict: 所有出现词的词频字典
        :return: 类与对应出现过的词列表的字典
        """

        total_word = total_dict.keys()
        class_word_pool = {}
        for class_name in self._class_name:
            class_word_pool.setdefault(class_name, set())

        for line in with_class_list:
            if line[0] in self._class_name:
                for word in line[1:]:
                    if word in total_word:
                        if word in self._repeat_class_word:
                            class_word_pool[line[0]].add('_'.join(word))
                        else:
                            class_word_pool[line[0]].add(word)

        return class_word_pool

    def _normalize_class(self, x, scale):
        """
        对词频进行标准化，使其具有相同的尺度([0,1]区间)
        :param x:
        :param scale: 数据标准差
        :return:
        """
        # if scale < 30:
        #     return 5*(2 ^ x)
        # if scale > 150:
        #     return x / math.log(10*x)
        # else:
        return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

    def _normalize_word(self, x, scale):
        """
        对词频进行标准化，使其具有相同的尺度([0,1]区间)
        :param x:
        :param scale: 数据标准差
        :return:
        """
        # if scale < 30:
        #     return 5*(2 ^ x)
        # if scale > 150:
        #     return x / math.log(10*x)
        # else:
        return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

    def export_json(self, class_word_pool, total_dict):
        """
        输出图数据的json格式
        :param class_word_pool: 类与对应出现过的词列表的字典
        :param total_dict: 所有词的词频字典
        :return: 图数据的json格式
        """
        value_data = np.array(list(total_dict.values()))
        scale = value_data.std()
        print('标准值：', scale)

        graph_nodes = total_dict.keys()  # node名称list
        graph_nodes_sizes = total_dict.values()
        graph_edge = [(class_name, word) for class_name, word_set in class_word_pool.items() for word in word_set]

        colors = [name for name, hexe in matplotlib.colors.cnames.items()]
        colors_json = [hexe for name, hexe in matplotlib.colors.cnames.items()]
        G = nx.Graph()
        G.add_nodes_from(graph_nodes, size=graph_nodes_sizes)
        G.add_edges_from(graph_edge)
        pos = nx.spring_layout(G, k=0.7, scale=10, iterations=50)

        dict_node = {'nodes': [], 'edges': []}
        for node, size in total_dict.items():
            ran = np.random.randint(0, len(colors))
            dict_node['nodes'].append(
                {'id': node, 'color': colors_json[ran], 'size': size, 'x': pos[node][0] * 1000, 'y': pos[node][1] * 1000})
        for t in graph_edge:
            dict_node['edges'].append({'from': t[0], 'to': t[1], 'size': 1})
        node_json = json.dumps(dict_node, ensure_ascii=False)

        return node_json


def process(data, word_threshold, class_threshold):

    start = time.time()

    # 将data放入设计的数据结构
    list_data = np.array(data)
    data = pd.DataFrame(list_data, columns=['icontitle', 'description'])

    # 列表每项首个单词为对应类别，用‘ ’连接
    gj = GraphJson(data, class_column='icontitle', description_column='description')
    with_class_list = gj.seg_and_rm_stopwords(seg_flags=['n', 'a'], stopwords_relative_pos='lib/stopwords_biaodian.txt')

    end1 = time.time()
    excape1 = end1-start
    print('IO花费的时间：', excape1)

    # 数据清洗，根据输入阈值过滤类别和分词，去除重复。（class空值如何处理）
    total_dict = gj.gen_total_dict(with_class_list=with_class_list, word_threshold=word_threshold, class_threshold=class_threshold)

    # 类与对应出现过的词列表的字典，用于图edge的生成
    class_word_pool = gj.gen_class_word_pool(with_class_list, total_dict)

    end2 = time.time()
    excape2 = end2-end1
    print('清洗花费的时间：', excape2)

    # 输出图的json数据
    graph_json = gj.export_json(class_word_pool, total_dict)

    end3 = time.time()
    excape3 = end3-end2
    print('制图花费的时间：', excape3)
    print('总时间：', end3-start)

    return graph_json







