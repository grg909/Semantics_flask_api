import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import collections
import numpy as np
import json
import jieba.posseg as pseg
import math
import time

# ws = os.getcwd()
# ws = "/var/www/FlaskApp/FlaskApp/semnetwork"

# 防止matplotlib依赖tkinder报错
matplotlib.use('Agg')

#  TODO : 可以将 removeStopWords 函数利用文件缓存数据的过程重构到内存中处理


def text_processing(word_threshold, class_threshold, data):
    # os.chdir(ws)

    # 第一步：读取路见评价，分词，去除停用词
    def iter_segment(data, flags):
        """
        描述分词，用‘ ’分隔，去除描述分词中和类别冲突的词，并用其类别词连接作为第一个词。生成器版本
        :param data: 数据pandas.dataframe, 有两列，第一列类别，第二列描述
        :param flags: 指定保留的分词flags列表
        :return: 返回一个可以输出每行分词结果（字符串）的迭代器
        """
        try:
            flags[0]
        except IndexError:
            print('请输入分词保留的flags列表')
            raise

        for row in data.itertuples(index=False):
            description_seg = pseg.cut(row[1].replace('\r\n', ''))
            words_seg = [word for word, flag in description_seg if flag in flags and word != row[0]]
            seg_line = row[0]+' '+' '.join(words_seg)
            yield seg_line

    def removeStopwords(seg_line, stopwords_relative_pos):
        """
        读取停用词列表，去除输入分词字符串中的停用词，生成器版本
        :param seg: 分词结果字符串
        :param stopwords_relative_pos: 停用词相对路径位置
        :return: 返回一个可以输出每行无停用词结果（字符串）的迭代器
        """

        try:
            with open(stopwords_relative_pos, encoding='utf-8') as sp:
                stopwords = sp.read()
                stopwords_line = stopwords.split('\n')
        except FileNotFoundError:
            print('请确保停用词目录')
            raise

        seg_list = seg_line.split(' ')
        without_stopwords = [word for word in seg_list if word not in stopwords_line]
        removed_line = ' '.join(without_stopwords)
        yield removed_line

    """ 第一步:读取路见评价，分词，在首行加入类型并去除停用词  """
    start = time.time()
    list_data = np.array(data)
    data = pd.DataFrame(list_data, columns=['icontitle', 'description'])
    class_items = [item for item, count in collections.Counter(data['icontitle']).items() if count > int(class_threshold)]

    clean_data = []
    for seg in iter_segment(data, ['a','n']):
        for seg_rm in removeStopwords(seg, 'lib/stopwords_biaodian.txt'):
            clean_data.append(seg_rm)

    with open('record4.txt', 'w', encoding='utf-8') as re:
        for i in clean_data:
            re.write(str(i))
            re.write('\n')

    end1 = time.time()
    excape1 = end1-start
    print('花费的时间：', excape1)

    """ 第二步:读取分词结果并list所有词 """

    ## 未解决的问题，如果大类与评价中分词结果有冲突怎么处理
    seen = set()
    for item in class_items:
        seen.add(item)
    uniq = list(seen)

    def word_filtration(threshold):
        pool = {}  # 统计每个词出现的频次
        pool_filtered = {}  # 筛选满足出现频次条件的词以及对应的频次
        global class_pool
        class_pool = {}  #类对应出现过的词


        for seg in clean_data:  # seg 每一行
            count2 = 0
            each_line = seg.split(' ')  # eachline 每一行的所有词
            for each_word in each_line[1:]:    # 每行除去首个class的word
                class_pool.setdefault(each_line[0], [])  #
                if each_word not in class_pool[each_line[0]]:   # 去除描述分词中和类别相同的词
                    class_pool[each_line[0]].append(each_word)
                if each_word not in seen:                      # 如果这个词第一次出现 放到seen 如果出现过，+1
                    seen.add(each_word)  # 总的词典
                else:
                    if each_word == '经济':
                        print(count2)
                    count2 += 1
                    pool[each_word] = count2

        with open('record5.txt', 'w', encoding='utf-8') as re:
            for i in pool.items():
                re.write(str(i))
                re.write('\n')

        for k, v in pool.items():
            if v > int(threshold):
                pool_filtered[k] = v  # 筛选过的词和频率
            else:
                continue

        return pool_filtered

    pool_filtered = word_filtration(int(word_threshold))

    a = sorted(pool_filtered.items(), key=lambda d: d[1])
    with open('record1.txt', 'w', encoding='utf-8') as re:
        for i in a:
            re.write(str(i))
            re.write('\n')

    end2 = time.time()
    excape2 = end2-start
    print('花费的时间：', excape2)

    """ 第三步:计算共现次数 """
    keys_filtered = list(pool_filtered.keys())

    totalpool = []  # edge起止点list
    totalnodes = []  # node名称list
    totalsizes = []  # node尺寸list
    pool_json = {}
    colors = [name for name, hexe in matplotlib.colors.cnames.items()]
    colors_json = [hexe for name, hexe in matplotlib.colors.cnames.items()]
    totalcolors = [color for color in colors if 'light' not in color]

    for k, v in class_pool.items():
        if k in class_items:  # 剔除数量过少的class
            totalnodes.append(k)
            totalsizes.append(len(v))  # 把过滤后的class放到node里
            for i in v:  # 生成edge起止点
                if i in keys_filtered:
                    totalpool.append((k, i))
        else:
            pass

    def normalization(nodesizes):
        return [2*i/math.log(i) for i in nodesizes]

    def normalization2(nodesizes):
         return 2*nodesizes/math.log(nodesizes)

    totalnodes.extend(pool_filtered.keys())
    totalsizes.extend(pool_filtered.values())
    totalsizes=normalization(totalsizes)

    for i in set(totalnodes):
        if i in class_items:
            pool_json[i] = len(class_pool[i])
        else:
            pool_json[i] = pool_filtered[i]

    end3 = time.time()
    excape3 = end3-start
    print('花费的时间：', excape3)

    def export_json():
        G = nx.Graph()
        G.add_nodes_from(totalnodes, size=totalsizes)
        G.add_edges_from(totalpool)
        plt.figure(figsize=(150, 150))
        pos = nx.spring_layout(G, k=0.1)
        nx.draw_networkx_nodes(G, pos, node_size=totalsizes, alpha=0.8)

        dict_node = {'nodes': [], 'edges': []}
        for n, m in pool_json.items():
            m = normalization2(m)
            ran = np.random.randint(0, len(colors))
            dict_node['nodes'].append(
                {'id': n, 'color': colors_json[ran], 'size': m, 'x': pos[n][0] * 1000, 'y': pos[n][1] * 1000})
        for t in totalpool:
            dict_node['edges'].append({'from': t[0], 'to': t[1], 'size': 1})
        node_json = json.dumps(dict_node, ensure_ascii=False)

        return node_json

    try:
        return export_json()
    finally:
            pass
