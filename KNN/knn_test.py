from KNN import knn_train, data_init
import numpy as np


def knn_test():
    train_x, train_y = data_init.data_init()

    test_x = np.array([[1.2, 1.0], [0.1, 0.3]])
    k = 3
    for x in test_x:
        test_label = knn_train.knn_classify(x, train_x, train_y, k)
        print("The input:", x, "is classify to ", test_label)


if __name__ == '__main__':
    knn_test()
