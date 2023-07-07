'''
主要更改内容:
    修正函数名和变量名的拼写错误。
    规范函数和变量命名，采用小写字母和下划线的组合，以增加可读性。
    删除不必要的空行和注释。
    调整代码缩进，使其符合PEP 8规范。
    使用更加直观的方式来构造列表和字符串。
'''
import pickle
from collections import Counter


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')


def single_list(arr, target):
    return arr.count(target)


# staqc: 将语料中的单候选和多候选分隔开
def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    with open(filepath, 'r') as f:
        total_data = eval(f.read())

    qids = [total_data[i][0][0] for i in range(len(total_data))]

    result = Counter(qids)

    total_data_single = []
    total_data_multiple = []

    for i in range(len(total_data)):
        if result[total_data[i][0][0]] == 1:
            total_data_single.append(total_data[i])
        else:
            total_data_multiple.append(total_data[i])

    with open(save_single_path, 'w') as f:
        f.write(str(total_data_single))

    with open(save_multiple_path, 'w') as f:
        f.write(str(total_data_multiple))


# large: 将语料中的单候选和多候选分隔开
def data_large_processing(filepath, save_single_path, save_multiple_path):
    total_data = load_pickle(filepath)
    qids = [total_data[i][0][0] for i in range(len(total_data))]

    result = Counter(qids)
    total_data_single = []
    total_data_multiple = []

    for i in range(len(total_data)):
        if result[total_data[i][0][0]] == 1:
            total_data_single.append(total_data[i])
        else:
            total_data_multiple.append(total_data[i])

    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)

    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


# 将单候选只保留其 qid
def single_unlabel2label(path1, path2):
    total_data = load_pickle(path1)
    labels = [[total_data[i][0], 1] for i in range(len(total_data))]

    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))

    with open(path2, 'w') as f:
        f.write(str(total_data_sort))


if __name__ == "__main__":
    # 将 staqc_python 中的单候选和多候选分开
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = '../hnn_process/ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    # data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    # 将 staqc_sql 中的单候选和多候选分开
    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = '../hnn_process/ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    # data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    # 将 large_python 中的单候选和多候选分开
    large_python_path = '../hnn_process/ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    # 将 large_sql 中的单候选和多候选分开
    large_sql_path = '../hnn_process/ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    # data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    large_sql_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single_label.txt'
    # single_unlabel2label(large_sql_single_save, large_sql_single_label_save)
    # single_unlabel2label(large_python_single_save, large_python_single_label_save)
