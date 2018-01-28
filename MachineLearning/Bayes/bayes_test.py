from Bayes import bayes_train, data_init
import numpy as np


def bayes_test():
    """
    Test function
    :return:
    """
    train_x_word, train_y = data_init.load_data()

    dic = data_init.create_dictionary(train_x_word)

    train_x_vec = []

    for x in train_x_word:
        train_x_vec.append(data_init.dic_vectoring(dic, x))

    p0_vec, p1_vec, p_insult = bayes_train.bayes_train(np.array(train_x_vec), np.array(train_y))

    test_x_words = [
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['love']
    ]

    for test_x_word in test_x_words:
        test_x_vec = data_init.dic_vectoring(dic, test_x_word)
        if bayes_train.classify(np.array(test_x_vec), p0_vec, p1_vec, p_insult):
            print(test_x_word, "is insulting.")
        else:
            print(test_x_word, "is not insulting.")


if __name__ == '__main__':
    bayes_test()
