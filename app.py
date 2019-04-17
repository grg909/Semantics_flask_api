from flask import render_template, request
import pymysql
import json
from .lib.wordnet_json import WordnetJson
from . import app
import numpy as np
import pandas as pd


class MysqlConn:

    @staticmethod
    def perform(sql):
        db = pymysql.connect(host='rm-2zeq1de3sv61152354o.mysql.rds.aliyuncs.com', user='root', password='Ctdna1903#10000',
                             db='pinstreet_app', port=3306, charset='utf8mb4')
        try:
            with db.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
            db.commit()
            return results
        except Exception as e:
            raise e

        finally:
            db.close()


@app.route('/get_json')
def get_json():
    class_threshold = request.args.get('class_threshold')
    word_threshold = request.args.get('word_threshold')
    ac_id = request.args.get('ac_id')

    sql = "SELECT fy_icon.icontitle, fy_proposal.description FROM fy_proposal LEFT JOIN fy_icon on " \
          "fy_proposal.icon_id = fy_icon.id WHERE fy_proposal.activity_id = {} AND fy_proposal.status = 1 and " \
          "fy_proposal.delete_status = 0 ;".format(ac_id)

    data = MysqlConn.perform(sql=sql)
    word_threshold, class_threshold = word_threshold, class_threshold

    # 将data放入设计的数据结构
    list_data = np.array(data)
    data = pd.DataFrame(list_data, columns=['icontitle', 'description'])

    wj = WordnetJson(data, class_column='icontitle', description_column='description', keywords_list=['建议'])

    graph_json = wj.gen_wordnet_json(seg_flags=['n', 'a'], stopwords_relative_pos='lib/stopwords_biaodian.txt')

    return json.dumps(graph_json)


@app.route('/index', methods=['POST', 'GET'])
def input_page():
    word_threshold = request.form['word_threshold']
    class_threshold = request.form['class_threshold']
    ac_id = request.form['ac_id']
    return render_template('collection_index.html', ac_id=ac_id, word_threshold=word_threshold, class_threshold=class_threshold)


@app.route('/')
def index_dc():
    sql = 'select id, title from fy_activity;'
    datas = MysqlConn.perform(sql=sql)
    if datas:
        activity = dict(datas)
        return render_template("navigation.html", activity=activity)
    else:
        return 'data_error'
