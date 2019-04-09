# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:06:08 2018

@author: ERT
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import os
import collections
# import jieba
import numpy as np
# import codecs
import json
import jieba.posseg as pseg

csv_name = "/home/yifeiLu/data_visualization/lib/fy_icon.csv"
ws = '/home/yifeiLu/data_visualization'

totalpool = []  # edge起止点list
totalnodes = []  # node名称list
totalsizes = []  # node尺寸list
pool_json = {}
colors = [name for name, hexe in matplotlib.colors.cnames.items()]
colors_json = [hexe for name, hexe in matplotlib.colors.cnames.items()]
totalcolors = [color for color in colors if 'light' not in color]
seen = set()


def export_json():
    G = nx.Graph()
    G.add_nodes_from(totalnodes, size=totalsizes)
    G.add_edges_from(totalpool)
    plt.figure(figsize=(30, 30))
    pos = nx.spring_layout(G)

    # nx.draw_networkx_nodes(G, pos, node_size=totalsizes, node_color=totalcolors, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, node_size=totalsizes, alpha=0.8)

    dict_node = {'nodes': [], 'edges': []}
    for n, m in pool_json.items():
        ran = np.random.randint(0, len(colors))
        dict_node['nodes'].append(
            {'id': n, 'color': colors_json[ran], 'size': m, 'x': pos[n][0] * 1000, 'y': pos[n][1] * 1000})
    # print(totalpool)
    for t in totalpool:
        dict_node['edges'].append({'from': t[0], 'to': t[1], 'size': 1})
        node_json = json.dumps(dict_node, ensure_ascii=False)
    # print(node_json)
    return node_json


def removeStopWords(seg_list):
    wordlist_stopwords_removed = []

    stop_words = open(ws + '/lib/stopwords_biaodian.txt', encoding='utf-8')
    stop_words_text = stop_words.read()

    stop_words.close()

    stop_words_text_list = stop_words_text.split('\n')
    after_seg_text_list = seg_list.split(' ')

    for word in after_seg_text_list:
        if word not in stop_words_text_list:
            wordlist_stopwords_removed.append(word)

    without_stopwords = open(ws + '/datafile/segment_wo_stopwords.txt', 'a')
    without_stopwords.write(' '.join(wordlist_stopwords_removed))
    without_stopwords.write('\n')
    return ' '.join(wordlist_stopwords_removed)


def segment(doc, class_name):
    doc_seg = pseg.cut(doc)
    words_seg = []
    for word, flag in doc_seg:
        if (flag in ['n', 'a']):
            words_seg.append(word)
    seg_list = " ".join(words_seg)  # seg_list为str类型
    document_after_segment = open('datafile/segment.txt', 'a', encoding='GB18030')
    document_after_segment.write(class_name)
    document_after_segment.write(' ')
    document_after_segment.write(seg_list)
    document_after_segment.write('\n')
    document_after_segment.close()
    return seg_list


def word_filtration(threshold):
    # global seen
    global class_pool
    global seen
    pool = {}  # 统计每个词出现的频次
    pool_filtered = {}  # 筛选满足出现频次条件的词以及对应的频次
    class_pool = {}  # 每个类对应出现过的词
    fpp = open('datafile/segment_wo_stopwords.txt', 'a')
    fpp.close()

    with open('datafile/segment_wo_stopwords.txt', encoding="utf-8") as fp:
        words_text = fp.read()
        seglist = words_text.split('\n')  # 每条分词后 共有6670行

        for seg in seglist[:-1]:  # seg 每一行
            count2 = 0
            each_line = seg.split(' ')  # eachline 每一行的所有词
            for each_word in each_line[1:]:
                class_pool.setdefault(each_line[0], [])  #
                if each_word not in class_pool[each_line[0]]:
                    class_pool[each_line[0]].append(each_word)
                if each_word not in seen:
                    seen.add(each_word)  # 总的词典
                else:
                    count2 += 1
                    pool[each_word] = count2

    for k, v in pool.items():
        if v > int(threshold):
            pool_filtered[k] = v  # 筛选过的词和频率
        else:
            continue

    return pool_filtered


def text_processing(word_threshold, class_threshold, data):
    print('进入函数')
    """ 第一步:读取路见评价，分词，在首行加入类型并去除停用词  """
    # data = list(data)
    # data.insert(0,[])
    os.chdir(ws)
    list_data = np.array(data)
    data = pd.DataFrame(list_data, columns=['icontitle', 'description'])
    # print(data)
    #
    # print('sss', collections.Counter(data['icontitle']).items())

    class_items = [item for item, count in collections.Counter(data['icontitle']).items() if
                   count > int(class_threshold)]
    # print('class_i', class_items)
    for index, row in data.iterrows():
        segment(row['description'].replace('\r\n', ''), row['icontitle'])

    with open('datafile/segment.txt', encoding='GB18030') as fp:
        words_text = fp.read()
        seglist = words_text.split('\n')
        for seg in seglist[:-1]:
            removeStopWords(seg)
    print('第一步=======')
    """ 第二步:读取分词结果并list所有词 """

    # TODO 未解决的问题，如果大类与评价中分词结果有冲突怎么处理

    for item in class_items:
        seen.add(item)
    uniq = list(seen)

    pool_filtered = word_filtration(int(word_threshold))
    # print('所有词list已建立完成。')
    print('第二步=====')
    """ 第三步:计算共现次数 """
    keys_filtered = list(pool_filtered.keys())
    # print('class_pool.items()', class_pool.items())
    # print('class_pool', class_pool)
    for k, v in class_pool.items():
        if k in class_items:  # 剔除数量过少的class
            totalnodes.append(k)
            totalsizes.append(len(v))
            for i in v:  # 生成edge起止点
                if i in keys_filtered:
                    totalpool.append((k, i))
        else:
            pass
    # print('共有edge {} 条。'.format(len(totalpool)))

    totalnodes.extend(pool_filtered.keys())
    totalsizes.extend(pool_filtered.values())

    for i in set(totalnodes):
        if i in class_items:
            pool_json[i] = len(class_pool[i])
        else:
            try:
                pool_json[i] = pool_filtered[i]
            except:
                pool_json[i] = 'key == null'
                # pass

    # plot_nx(totalpool)  # 图形生成
    print('第三步========')
    try:
        return export_json()

    finally:
        try:
            os.remove(ws + '/datafile/segment.txt')
            os.remove(ws + '/datafile/segment_wo_stopwords.txt')
        except:
            pass


if __name__ == '__main__':
    word_threshold = input('设置词语最小出现次数：')
    class_threshold = 10  # 多于这个threshold的class才计入有效数据
    os.chdir(ws)
    text_processing(word_threshold, class_threshold)
