from flask import render_template, request
from database import MysqlConn
import json
from .lib.wordnet_json import WordnetJson
import numpy as np
import pandas as pd
from flask import Flask

app = Flask(__name__)


@app.route('/get_json')
def get_json():
    class_threshold = request.args.get('class_threshold')
    word_threshold = request.args.get('word_threshold')
    auto = int(request.args.get('auto'))
    ac_id = request.args.get('ac_id')
    sql = "SELECT fy_icon.icontitle, fy_proposal.description FROM fy_proposal LEFT JOIN fy_icon on " \
          "fy_proposal.icon_id = fy_icon.id WHERE fy_proposal.activity_id = {} AND fy_proposal.status = 1 and " \
          "fy_proposal.delete_status = 0 ;".format(ac_id)

    data = MysqlConn.perform(sql=sql)

    # 将data放入设计的数据结构
    list_data = np.array(data)
    data = pd.DataFrame(list_data, columns=['icontitle', 'description'])

    wj = WordnetJson(data, data_name=ac_id, class_column='icontitle', description_column='description', keywords_list=['建议'])
    if auto == 1:
        graph_json = wj.gen_wordnet_json(seg_flags=['n', 'a'], stopwords_relative_pos='lib/hlt_stop_words.txt')
    else:
        graph_json = wj.gen_wordnet_json(seg_flags=['n', 'a'], stopwords_relative_pos='lib/hlt_stop_words.txt', word_threshold=word_threshold, class_threshold=class_threshold)

    return json.dumps(graph_json)


@app.route('/index', methods=['POST', 'GET'])
def input_page():
    word_threshold = request.form['word_threshold']
    class_threshold = request.form['class_threshold']
    ac_id = request.form['ac_id']
    auto = 0 if request.form.get('auto_mode') is None else 1
    return render_template('collection_index.html', ac_id=ac_id, word_threshold=word_threshold, class_threshold=class_threshold, auto=auto)


@app.route('/')
def index_dc():
    sql = 'select id, title from fy_activity WHERE status = 1 ORDER BY id DESC ;'
    datas = MysqlConn.perform(sql=sql)
    if datas:
        activity = dict(datas)
        return render_template("navigation.html", activity=activity)
    else:
        return 'data_error'
