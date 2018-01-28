from math import log
import operator


def calc_shannon_ent(dataset):
    """
    计算信息熵
    :param dataset: 数据集
    :return: 信息熵
    """
    num_entries = len(dataset)
    # 创建字典保存label以及其出现的次数
    label_count = {}

    for feature_vec in dataset:
        # 统计标签以及其数量
        label = feature_vec[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1

    # 计算信息熵
    shannon_ent = 0.0
    for key in label_count:
        prob = float(label_count[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)

    return shannon_ent


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                # [1, 1, 'maybe'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    """
    按照给定的特征划分数据集，即返回第axis个特征的值为value的数据项
    :param data_set: 待划分数据集
    :param axis: 给定的特征
    :param value: 需要返回的特征的值
    :return:
    """
    ret_data_set = []
    for feature_vec in data_set:
        if feature_vec[axis] == value:
            reduced_feature_vec = feature_vec[:axis]
            """
            a = [1, 2, 3]
            b = [4, 5, 6]
            a.append(b)
            a = [1, 2, 3, [4, 5, 6]]
            
            a.extend(b)
            a = [1, 2, 3, 4, 5, 6]
            """
            reduced_feature_vec.extend(feature_vec[axis + 1:])
            ret_data_set.append(reduced_feature_vec)
    return ret_data_set


def choose_best_feature(data_set):
    """
    选择最佳分类特征，使信息增益最大
    :param data_set: 数据集
    :return: best_feature: 最佳特征 best_info_gain: 最大信息增益
    """
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0  # 信息熵增益
    best_feature = -1  # 最佳判决特征

    for i in range(num_features):
        feature_list = [example[i] for example in data_set]
        unique_feature = set(feature_list)
        new_entropy = 0.0
        for value in unique_feature:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        # 计算信息增益
        info_gain = base_entropy - new_entropy

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
            # print("info_gain = ", info_gain, " best_feature = ", i)

    return best_feature, best_info_gain


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat, _ = choose_best_feature(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])
    feat_values =[example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


if __name__ == '__main__':
    data_set, labels = create_data_set()
    # print(calc_shannon_ent(data_set))
    # choose_best_feature(data_set)
    print(create_tree(data_set, labels))
