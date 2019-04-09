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
        :param scale: 1-10，1代表1k级别的数据，2代表10k
        """
        self.data = data
        self.class_column = class_column
        self.description_column = description_column
        self.scale = 1000

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
            description_seg = pseg.cut(row[1].replace('\r\n', ''))
            words_seg = [word for word, flag in description_seg if flag in flags and word != row[0]]
            seg_line = row[0]+' '+' '.join(words_seg)
            yield seg_line

    @staticmethod
    def _iter_remove_stopwords(seg_line, stopwords_relative_pos):
        """
        读取停用词列表，去除输入分词字符串中的停用词。生成器版本
        :param seg_line: 分词结果字符串
        :param stopwords_relative_pos: 停用词相对路径位置
        :return: 返回一个可以输出每行无停用词结果（字符串）的迭代器
        """
        try:
            with open(stopwords_relative_pos, encoding='utf-8') as sp:
                stopwords = sp.read()
                stopwords_line = stopwords.split('\n')
        except Exception:
            print('请确保停用词库在正确目录下')
            exit()

        seg_list = seg_line.split(' ')
        without_stopwords = [word for word in seg_list if word not in stopwords_line]
        removed_line = ' '.join(without_stopwords)
        yield removed_line

    def seg_and_rm_stopwords(self, seg_flags, stopwords_relative_pos):
        """
        一步执行分词和去除停用词
        :param seg_flags: 指定保留的分词flags列表
        :param stopwords_relative_pos: 停用词相对路径位置
        :return: 返回一个列表，每项为字符串，每个字符串为类别词加分词结果，用‘ ’连接
        """
        with_class_data = []
        for seg in self._iter_segment(seg_flags):
            for seg_rm in self._iter_remove_stopwords(seg, stopwords_relative_pos):
                with_class_data.append(seg_rm)

        return with_class_data

    def _class_clean(self, class_threshold):
        """
        去重以及筛选满足阈值的类别
        :param class_threshold: 类别频率阈值
        :return: 清洗后的类别列表
        """
        class_cleaned = {class_name: count for class_name, count in collections.Counter(self.data[self.class_column]).items() if count > int(class_threshold)}
        return class_cleaned

    @staticmethod
    def _word_clean(with_class_list, word_threshold):
        """
        统计分词词频并筛选满足阈值的类别。
        :param with_class_list: 描述分词和去停用词后得到的字符串列表
        :param word_threshold: 分词频率阈值
        :return: 分词及其词频字典
        """
        without_class_list = []
        for line in with_class_list:
            with_class_word = line.split(' ')
            without_class_list.append(with_class_word[1:])

        data = list(itertools.chain(*without_class_list))
        word_pool = {word: count for word, count in collections.Counter(data).items() if count > int(word_threshold)}
        return word_pool

    def gen_total_dict(self, with_class_list, word_threshold, class_threshold):
        """
        生成所有出现词的词频字典
        :param with_class_list: 描述分词和去停用词后得到的字符串列表
        :param word_threshold: 分词频率阈值
        :param class_threshold: 类别频率阈值
        :return: 所有出现词的词频字典
        """
        class_dict = self._class_clean(class_threshold)
        word_dict = self._word_clean(with_class_list, word_threshold)

        class_set = set(class_dict.keys())
        word_set = set(word_dict.keys())
        for repeat_name in class_set & word_set:
            word_dict['_'.join(repeat_name)] = word_dict.pop(repeat_name)
        total_dict = {**word_dict, **class_dict}
        return total_dict

    @staticmethod
    def gen_class_word_pool(with_class_list, total_dict):
        """
        生成类与对应出现过的词列表的字典
        :param with_class_list: 描述分词和去停用词后得到的字符串列表
        :param total_dict: 所有出现词的词频字典
        :return: 类与对应出现过的词列表的字典
        """
        seen = set()
        for item in total_dict.keys():
            if item not in seen:
                seen.add(item)

        class_word_pool = {}

        for row in with_class_list:
            line = row.split(' ')
            if line[0] in seen:
                for word in line[1:]:
                    if word in seen:
                        class_word_pool.setdefault(line[0], [])
                        if word not in class_word_pool[line[0]]:
                            class_word_pool[line[0]].append(word)

        return class_word_pool

    def normalize(self, x, scale):
        """
        根据数据规模不同，对分词词频进行不同程度的标准化，使得其具有相同的尺度
        :param x:
        :return:
        """
        if scale < 30:
            return 5*(2 ^ x)
        if scale > 150:
            return x / math.log(x)
        else:
            return 1.5*math.atan(x)*x/math.pi

    def export_json_echarts(self, class_word_pool, total_dict):
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
        graph_nodes_sizes = list(map(lambda x: self.normalize(x, scale), total_dict.values()))
        graph_edge = [(class_name, word) for class_name, word_list in class_word_pool.items() for word in word_list]

        colors = [name for name, hexe in matplotlib.colors.cnames.items()]
        colors_json = [hexe for name, hexe in matplotlib.colors.cnames.items()]
        G = nx.Graph()
        G.add_nodes_from(graph_nodes, size=graph_nodes_sizes)
        G.add_edges_from(graph_edge)
        pos = nx.spring_layout(G, scale=10, iterations=5)

        dict_node = {'nodes': [], 'edges': []}
        for node, size in total_dict.items():
            size = self.normalize(size, scale)
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
    with_class_list = gj.seg_and_rm_stopwords(seg_flags=['n', 'ns', 'a', 'an'], stopwords_relative_pos='lib/stopwords_biaodian.txt')

    end1 = time.time()
    excape1 = end1-start
    print('IO花费的时间：', excape1)

    # 数据清洗，根据输入阈值过滤类别和分词，去除重复。（class空值如何处理）
    total_dict = gj.gen_total_dict(with_class_list=with_class_list, word_threshold=word_threshold, class_threshold=class_threshold)

    # 类与对应出现过的词列表的字典，用于图edge的生成
    class_word_pool = gj.gen_class_word_pool(with_class_list, total_dict)

    end2 = time.time()
    excape2 = end2-start
    print('清洗花费的时间：', excape2)

    # 输出图的json数据
    graph_json = gj.export_json_echarts(class_word_pool, total_dict)

    end3 = time.time()
    excape3 = end3-start
    print('制图花费的时间：', excape3)

    return graph_json







