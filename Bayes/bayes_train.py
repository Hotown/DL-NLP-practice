import numpy as np


def bayes_train(train_input, train_label):
    """
    Train function
    :param train_input: train_x
    :param train_label: train_y
    :return:
    """
    input_size = len(train_input)
    x_size = len(train_input[0])

    p_insults = sum(train_label) / float(input_size)  # the probability of insulting sentence
    p0_num = np.ones(x_size)
    p1_num = np.ones(x_size)
    p0_demon = 2.0
    p1_demon = 2.0
    for i in range(input_size):
        if train_label[i] == 1:
            p1_num += train_input[i]
            p1_demon += sum(train_input[i])
        else:
            p0_num += train_input[i]
            p0_demon += sum(train_input[i])
    p0_vec = np.log(p0_num / p0_demon)
    p1_vec = np.log(p1_num / p1_demon)
    return p0_vec, p1_vec, p_insults


def classify(input_vector, p0_vec, p1_vec, p_insults):
    """
    Classify function
    :param input_vector: vector to classify
    :param p0_vec: not insult
    :param p1_vec: insult
    :param p_insults: the probability of insult
    :return:
    """
    p0 = sum(input_vector * p0_vec) + np.log(p_insults)  # log(A*B) = logA + logB
    p1 = sum(input_vector * p1_vec) + np.log(1 - p_insults)

    if p1 < p0:
        return 1
    else:
        return 0
