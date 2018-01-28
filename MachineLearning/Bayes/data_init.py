# -*- coding: UTF-8 -*-


def load_data():
    """
    Testing data set
    1. My dog has flea problems, help please! => 0
    2. Maybe not take him to dog park, stupid. => 1
    3. My dalmation is so cute, I love him. => 0
    4. Stop posting stupid worthless garbage. => 1
    5. Mr licks ate my steak, how to stop him. => 0
    6. Quit buying worthless dog food, stupid. => 1
    :return:
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vector = [0, 1, 0, 1, 0, 1]  # 1 means insulting sentence, 0 means not
    return posting_list, class_vector


def create_dictionary(data_set):
    """
    Create a dictionary including all the word in the test data set.
    :param data_set:
    :return: list
    """
    dictionary = set([])
    for sentence in data_set:
        dictionary = dictionary | set(sentence)
    return list(dictionary)


def dic_vectoring(dictionary, input_set):
    """
    Vectoring the word.
    :param dictionary: the dictionary returned in created_dictionary
    :param input_set: input set
    :return: word_vector
    """
    dic_vector = [0] * len(dictionary)
    for word in input_set:
        if word in dictionary:
            dic_vector[dictionary.index(word)] = 1
        else:
            print("Word %s is not in the dictionary." % word)
    return dic_vector
