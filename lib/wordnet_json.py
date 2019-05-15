# coding: utf-8

# @Time    : 2019/04/11
# @Author  : WANG JINGE
# @Site    :
# @File    : wordnet_json.py
# @Software: PyCharm
"""
    此类用于文本分析
"""

from collections import Counter
from itertools import chain
from json import dumps

import matplotlib
import networkx as nx
from jieba import posseg as pseg
import numpy as np
import pickle
import csv

# TODO ： 1. 容错机制 2. 可读性 3. 性能优化


class WordnetJson:

    def __init__(
            self,
            data,
            class_column,
            description_column,
            keywords_list=None):
        """
        传入dataframe
        :param data: 数据pandas.dataframe, 有两列，第一列类别，第二列描述
        :param class_column: 定义的类别列名
        :param description_column: 定义的描述列名
        :param keywords_list: 用户自定义的需提高识别度（node大小权重）的关键词list
        """
        self.data = data
        self.class_column = class_column
        self.description_column = description_column
        self.keywords_list = keywords_list

        # 词频大小标准化时的调整参数，使类别词节点与分词节点明显区分
        self.class_nor_coefficient = 80
        self.class_nor_offset = 0.3
        self.word_nor_coefficient = 35
        self.word_nor_offset = 0.15

        # 为了默认出图显示效果，分词只取词频大小排序前110个
        self.word_limitation = 110

        self.__repeat_class_word = set()
        self.__class_name = []
        self.__uncleaned_word_list = []
        self.__keywords_existed = []
        self.__keywords_dict = {}
        self.__keywords_nor = {}

    def __iter_segment(self, flags):
        """
        描述分词，用‘ ’分隔，去除描述分词中和类别冲突的词，并用其类别词连接作为第一个词。生成器
        :param flags: 指定保留的分词flags列表
        :return: 返回一个可以输出每行分词结果（字符串）的迭代器
        """
        for row in self.data.itertuples(index=False):
            seg_list = []
            description_seg = pseg.cut(row[1].replace('\r\n', ''))
            words_seg = [
                word for word,
                flag in description_seg if flag in flags and word != row[0] and len(word) > 1]
            seg_list.append(row[0])
            seg_list.extend(words_seg)
            yield seg_list

    @staticmethod
    def __iter_remove_stopwords(seg_list, stopwords_line):
        """
        读取停用词列表，去除输入分词字符串中的停用词。生成器
        :param seg_list: 分词结果字符串
        :param stopwords_line: 停用词列表
        :return: 返回一个可以输出每行无停用词结果（字符串）的迭代器
        """
        for word in seg_list:
            if word not in stopwords_line:
                yield word

    def seg_and_rm_stopwords(self, seg_flags, stopwords_relative_pos):
        """
        分词和去除停用词
        :param seg_flags: 指定保留的分词flags列表
        :param stopwords_relative_pos: 停用词相对路径位置
        :return: 返回一个列表，每项为字符串，每个字符串为类别词加分词结果，用‘ ’连接
        """
        try:
            with open(stopwords_relative_pos, encoding='utf-8') as sp:
                stopwords = sp.read()
                stopwords_line = stopwords.split('\n')
        except Exception as e:
            print('请确保停用词库在正确目录下')
            raise e

        with_class_data = []
        seg_list = self.__iter_segment(seg_flags)
        seg_rm = self.__iter_remove_stopwords(seg_list, stopwords_line)
        for line in seg_rm:
            with_class_data.append(line)

        return with_class_data

    @staticmethod
    def __word_threshold_calculation(word_count):
        """
        计算分词的默认阈值参数
        :param word_count: 分词词频字典
        :return: 分词阈值
        """
        word_count_list = list(word_count.values())
        word_count_array = np.array(word_count_list)

        word_threshold = np.percentile(word_count_array, 80)

        print('word_threshold: ', word_threshold)
        return word_threshold

    @staticmethod
    def __class_threshold_calculation(class_count):
        """
        计算类别词的默认阈值参数
        :param class_count: 类别词词频字典
        :return: 类别词阈值
        """
        class_count_list = list(class_count.values())
        word_count_array = np.array(class_count_list)
        std = np.std(word_count_array)
        if std < 500:
            class_threshold = np.percentile(word_count_array, 30)
        else:
            class_threshold = np.percentile(word_count_array, 60)

        print('class_threshold: ', class_threshold)
        return class_threshold

    def __class_normalise(self, class_cleaned):
        """
        类别词词频的标准化及调整
        :param class_cleaned: 类别词词频字典
        :return: 调整词频后的类别词字典
        """
        try:
            max_count = max(class_cleaned.values())
            min_count = min(class_cleaned.values())
        except Exception as e:
            print('类别数据为空')
            raise e

        try:
            for class_name, count in class_cleaned.items():
                class_cleaned[class_name] = self.class_nor_coefficient * (
                    self.class_nor_offset + ((float(count) - min_count) / float(max_count - min_count)))
        except Exception as ex:
            print('类别数据标准化出错')
            raise ex

        return class_cleaned

    def __word_normalise(self, word_pool):
        """
        分词词频的标准化及调整
        :param word_pool: 分词词频字典
        :return: 调整词频后的分词词频字典
        """
        try:
            max_count = max(word_pool.values())
            min_count = min(word_pool.values())
        except Exception as e:
            print('分词数据为空')
            raise e

        try:
            for word_name, count in word_pool.items():
                word_pool[word_name] = self.word_nor_coefficient * (self.word_nor_offset + (
                    (float(count) - min_count) / float(max_count - min_count)))
        except Exception as ex:
            print('分词数据标准化出错')
            raise ex

        return word_pool

    def __class_clean(self, class_threshold=None):
        """
        去重以及筛选满足阈值的类别，词频数据标准化与调整
        :param class_threshold: 类别词词频阈值
        :return: 清洗后的类别词字典
        """
        if class_threshold is None:
            class_count = {class_name: count for class_name, count in Counter(
                self.data[self.class_column]).items()}
            # 计算默认的类别词阈值
            class_threshold = self.__class_threshold_calculation(class_count)
            class_cleaned = {
                word: count for word,
                count in class_count.items() if count > int(class_threshold)}
        else:
            class_cleaned = {class_name: count for class_name, count in Counter(
                self.data[self.class_column]).items() if count > int(class_threshold)}

        class_nor = self.__class_normalise(class_cleaned)

        return class_nor

    def __word_clean(self, with_class_list, word_threshold=None):
        """
        统计分词词频并筛选满足阈值的类别。
        :param with_class_list: 描述分词和去停用词后得到的字符串列表
        :param word_threshold: 分词频率阈值
        :return: 分词及其词频字典
        """
        without_class_list = []
        for line in with_class_list:
            without_class_list.append(line[1:])
        self.__uncleaned_word_list = list(chain(*without_class_list))

        if word_threshold is None:
            word_count = {word: count for word, count in Counter(
                self.__uncleaned_word_list).items()}
            # 计算默认的分词阈值
            word_threshold = self.__word_threshold_calculation(word_count)
            word_pool_cal = {
                word: count for word,
                count in word_count.items() if count > int(word_threshold)}

            # 为了保证默认出图的视觉效果，仅保留词频排序前110个分词
            word_pool_limited = sorted(
                word_pool_cal.items(),
                key=lambda t: t[1],
                reverse=True)[
                :self.word_limitation]
            word_pool = dict(word_pool_limited)
        else:
            word_pool = {word: count for word, count in Counter(
                self.__uncleaned_word_list).items() if count > int(word_threshold)}

        word_pool_nor = self.__word_normalise(word_pool)

        # 关键词的词频大小单独调整
        if self.keywords_list is not None:
            keywords_set = set(self.keywords_list)
            word_set = set(word_pool.keys())
            self.__keywords_existed = keywords_set & word_set
            if len(self.__keywords_existed):
                for keywords in self.__keywords_existed:
                    if word_pool_nor[keywords] < self.class_nor_offset:
                        word_pool_nor[keywords] += self.class_nor_offset


        return word_pool_nor

    def gen_total_dict(
            self,
            with_class_list,
            word_threshold=None,
            class_threshold=None):
        """
        生成所有词的词频字典（描述分词和类别词），若当前类别分词中出现和其他类别相同的名称，则分词中的用‘_'分隔重命名
        :param with_class_list: 描述分词和去停用词后得到的字符串列表
        :param word_threshold: 分词频率阈值
        :param class_threshold: 类别频率阈值
        :return: 所有出现词的词频字典
        """
        word_dict = self.__word_clean(with_class_list, word_threshold)
        class_dict = self.__class_clean(class_threshold)

        # 获取类别和分词中的相同名称列表，对应分词中用‘_'分隔每个字进行重命名，避免冲突
        class_set = set(self.data[self.class_column])
        word_set = set(self.__uncleaned_word_list)
        self.__repeat_class_word = class_set & word_set
        for repeat_name in self.__repeat_class_word:
            if repeat_name in word_dict:
                word_dict['_'.join(repeat_name)] = word_dict.pop(repeat_name)

        self.__class_name = class_dict.keys()
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
        class_word_pool = {class_name: set()
                           for class_name in self.__class_name}

        for line in with_class_list:
            if line[0] in self.__class_name:
                for word in line[1:]:
                    if word in total_word:
                        if word in self.__repeat_class_word:
                            class_word_pool[line[0]].add('_'.join(word))
                        else:
                            class_word_pool[line[0]].add(word)

        return class_word_pool

    @staticmethod
    def __parameter_calculation(scale, class_num):
        """
        根据数据量计算networkx.spring_layout函数的参数
        :param scale: 总的单词量
        :param class_num: 类别的数量
        :return: spring_layout参数k，iteration
        """
        print(scale)
        print(class_num)
        if scale < 30:
            k_cal = 0.2
        else:
            k_cal = 0.25 + scale / 363

        if class_num <= 5:
            iter_cal = 50
        elif scale < 150:
            iter_cal = 20
        else:
            iter_cal = 15

        return k_cal, iter_cal

    def export_json(self, class_word_pool, total_dict):
        """
        输出生成图的json数据
        :param class_word_pool: 类与对应出现过的词列表的字典
        :param total_dict: 所有词的词频字典
        :return: 生成图的json数据
        """
        graph_nodes = total_dict.keys()  # node名称list
        graph_nodes_sizes = total_dict.values()
        graph_edge = [(class_name, word) for class_name,
                      word_set in class_word_pool.items() for word in word_set]

        colors = [name for name, hexe in matplotlib.colors.cnames.items()]
        colors_json = [hexe for name, hexe in matplotlib.colors.cnames.items()]
        G = nx.Graph()
        G.add_nodes_from(graph_nodes, size=graph_nodes_sizes)
        G.add_edges_from(graph_edge)

        # 根据数据量计算制图参数
        scale = len(total_dict)
        class_num = len(class_word_pool.keys())
        fixed_class_nodes = list(graph_nodes)[-class_num:]
        fixed_nodes = list(chain(fixed_class_nodes, self.__keywords_existed))
        k_cal, iter_cal = self.__parameter_calculation(scale, class_num)
        pos = nx.spring_layout(
            G,
            k=k_cal,
            fixed=fixed_nodes,
            iterations=iter_cal)

        dict_node = {'nodes': [], 'edges': []}
        for node, size in total_dict.items():
            ran = np.random.randint(0, len(colors))
            dict_node['nodes'].append({'id': node,
                                       'color': colors_json[ran],
                                       'size': size,
                                       'x': pos[node][0] * 1000,
                                       'y': pos[node][1] * 1000})
        for t in graph_edge:
            dict_node['edges'].append({'from': t[0], 'to': t[1], 'size': 1})
        graph_json = dumps(dict_node, ensure_ascii=False)

        return graph_json

    def gen_wordnet_json(
            self,
            seg_flags,
            stopwords_relative_pos,
            word_threshold=None,
            class_threshold=None):
        """
        实现数据处理到最终输出json的总接口函数
        :param seg_flags: 指定保留的分词flags列表
        :param stopwords_relative_pos: 停用词相对路径位置
        :param word_threshold: 分词频率阈值
        :param class_threshold: 类别频率阈值
        :return: 生成图的json数据
        """
        with_class_list = self.seg_and_rm_stopwords(
            seg_flags, stopwords_relative_pos)

        # 数据清洗，根据输入阈值过滤类别和分词，去除重复。
        total_dict = self.gen_total_dict(
            with_class_list, word_threshold, class_threshold)

        # 类与对应出现过的词列表的字典，用于图edge的生成
        class_word_pool = self.gen_class_word_pool(with_class_list, total_dict)

        # 输出图的json数据
        graph_json = self.export_json(class_word_pool, total_dict)

        return graph_json

    def dict_to_csv(self, dict_data):
        """

        :param dict:
        :return:
        """
        with open('record.csv', 'wb', newline='') as f:
            w = csv.writer(f)
            w.writerows(dict_data.items())


