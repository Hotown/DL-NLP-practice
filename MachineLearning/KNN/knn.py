import numpy as np
import operator
from os import listdir
from collections import Counter


def file2matrix(filename):
    """
    将数据文件转化成矩阵
    :param filename: 数据文件路径
    :return: 数据矩阵，类别向量
    """
    fr = open(filename)

    # 获取文件行数
    number_of_lines = len(fr.readlines())
    # 创建初始空矩阵
    return_matrix = np.zeros((number_of_lines, 3))
    label_vector = []

    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # 分割字符串，去掉首尾空格
        list_from_line = line.strip().split('\t')
        return_matrix[index, :] = list_from_line[0:3]
        label_vector.append(list_from_line[-1])
        index += 1
    return return_matrix, label_vector


def auto_norm(dataset):
    """
    归一化
    :param dataset: 数据集
    :return: 完成归一化后的数据集
    """
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    range = max_vals - min_vals

    norm_dataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    #print(m)
    # 计算归一化矩阵 Y = (X - Xmin) / (Xmax - Xmin)
    norm_dataset = dataset - np.tile(min_vals, (m, 1))
    norm_dataset = norm_dataset / np.tile(range, (m, 1))
    return norm_dataset


def classify0(inX, dataset, labels, k):
    """

    :param inX: 输入向量
    :param dataset: 数据集
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return:
    """
    
    # 1. 计算距离
    dataset_size = dataset.shape[0]

    diff_mat = np.tile(inX, (dataset_size, 1)) - dataset
    # 欧式距离计算
    # 取平方
    sq_diff_mat = diff_mat ** 2
    # 行求和
    sq_distances = sq_diff_mat.sum(axis=1)
    # 开方
    distance = sq_distances ** 0.5
    sorted_distance = distance.argsort()

    # 2. 选择距离最小的k个点
    class_count = {}
    for i in range(k):
        # 找到该样本点的lable
        vote_ilabel = labels[sorted_distance[i]]
        # 在字典中加1
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1

    # 3. 排序，并返回出现次数最多的label
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def dating_class_test():
    # 测试集比例
    dev_percentage = 0.1
    # 加载数据
    dating_data_mat, dating_labels = file2matrix('data_set.txt')
    # 归一化
    norm_mat = auto_norm(dating_data_mat)
    # 设置测试样本数量
    m = norm_mat.shape[0]
    num_test = int(m * dev_percentage)
    error_count = 0.0
    for i in range(num_test):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test:m, :], dating_labels[num_test:m], 3)
        if classifier_result != dating_labels[i]:
            error_count += 1.0

    print("the total error count is: %f" % (error_count / float(num_test)))
    print(error_count)


if __name__ == '__main__':
    dating_class_test()
