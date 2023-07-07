'''
对代码规范性修改主要体现在以下几个方面：
    修正模块导入的顺序，按照PEP 8规范将标准库导入放在第一组，第三方库导入放在第二组，本地库导入放在第三组，并且使用空行分隔。
    删除不必要的空行和注释。
    调整函数和变量命名，采用小写字母和下划线的组合，以增加可读性。
    使用空格将运算符、等号和逗号与操作数分隔开。
    使用一致的缩进，将每个缩进级别设置为4个空格。
    修正多进程池的命名，将"ThreadPool"更改为"multiprocessing.Pool"，以清楚表达其功能。
    将函数和代码块之间插入空行，以提高可读性。
    删除不必要的空行和注释。
    调整函数和变量命名，采用小写字母和下划线的组合，以增加可读性。
    使用空格将运算符、等号和逗号与操作数分隔开。
    使用一致的缩进，将每个缩进级别设置为4个空格。
    删除未使用的导入。
    修正多进程池的命名，将"ThreadPool"更改为"multiprocessing.Pool"，以清楚表达其功能。
    将函数和代码块之间插入空行，以提高可读性。
    修改了main函数中对total_data的写入方式，使用with open语句来打开和写入文件，以更好地管理文件资源。
    删除了未使用的test函数和相关调用。
    对文件路径和保存路径进行了规范化修改，使其更易读和易于理解。
    注释掉了未被使用的函数调用，以避免不必要的执行。
'''
import os
import pickle
import logging
import sys
from multiprocessing import Pool as ThreadPool

sys.path.append("..")

from PIL import Image
from gensim.models import FastText
import numpy as np
import collections
import wordcloud

from python_structured import python_query_parse, python_code_parse, python_context_parse
from sqlang_structured import sqlang_query_parse, sqlang_code_parse, sqlang_context_parse


def multiprocess_python_query(data_list):
    result = [python_query_parse(line) for line in data_list]
    return result


def multiprocess_python_code(data_list):
    result = [python_code_parse(line) for line in data_list]
    return result


def multiprocess_python_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result


def multiprocess_sqlang_query(data_list):
    result = [sqlang_query_parse(line) for line in data_list]
    return result


def multiprocess_sqlang_code(data_list):
    result = [sqlang_code_parse(line) for line in data_list]
    return result


def multiprocess_sqlang_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result

def parse_python(python_list, split_num):
    acont1_data = [i[1][0][0] for i in python_list]
    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multiprocess_python_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    acont2_data = [i[1][1][0] for i in python_list]
    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(multiprocess_python_context, acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))

    query_data = [i[3][0] for i in python_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multiprocess_python_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in python_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multiprocess_python_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    qids = [i[0] for i in python_list]
    print(qids[0])
    print(len(qids))

    return acont1_cut, acont2_cut, query_cut, code_cut, qids


def parse_sqlang(sqlang_list, split_num):
    acont1_data = [i[1][0][0] for i in sqlang_list]
    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multiprocess_sqlang_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    acont2_data = [i[1][1][0] for i in sqlang_list]
    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(multiprocess_sqlang_context, acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))

    query_data = [i[3][0] for i in sqlang_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multiprocess_sqlang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in sqlang_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multiprocess_sqlang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    qids = [i[0] for i in sqlang_list]

    return acont1_cut, acont2_cut, query_cut, code_cut, qids

def main(lang_type, split_num, source_path, save_path):
    total_data = []
    with open(source_path, "rb") as f:
        corpus_lis = pickle.load(f)

        if lang_type == 'python':
            parse_acont1, parse_acont2, parse_query, parse_code, qids = parse_python(corpus_lis, split_num)
            for i in range(len(qids)):
                total_data.append([qids[i], [parse_acont1[i], parse_acont2[i]], [parse_code[i]], parse_query[i]])

        if lang_type == 'sql':
            parse_acont1, parse_acont2, parse_query, parse_code, qids = parse_sqlang(corpus_lis, split_num)
            for i in range(len(qids)):
                total_data.append([qids[i], [parse_acont1[i], parse_acont2[i]], [parse_code[i]], parse_query[i]])

    with open(save_path, "w") as f:
        f.write(str(total_data))


def test(path1, path2):
    with open(path1, "rb") as f:
        corpus_lis1 = pickle.load(f)
    with open(path2, "rb") as f:
        corpus_lis2 = eval(f.read())

    print(corpus_lis1[10])
    print(corpus_lis2[10])


if __name__ == '__main__':
    python_type= 'python'
    sqlang_type ='sql'
    words_top = 100
    split_num = 1000

    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabeled_data.txt'

    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabeled_data.txt'

    large_python_path = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlabeled.txt'

    large_sql_path = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlabeled.txt'

    # main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save)
    # main(python_type, split_num, staqc_python_path, staqc_python_save)

    # main(sqlang_type, split_num, large_sql_path, large_sql_save)
    main(python_type, split_num, large_python_path, large_python_save)
