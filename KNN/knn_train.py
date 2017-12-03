import numpy as np


def knn_classify(input_x, train_x, train_y, k):
    x_size = len(train_x)

    diff = np.tile(input_x, (x_size, 1)) - train_x
    squared_diff = diff ** 2
    squared_dist = np.sum(squared_diff, axis=1)
    distance = squared_dist ** 0.5

    sorted_dist_indices = np.argsort(distance)

    class_count = {}
    for i in range(k):
        vote_label = train_y[sorted_dist_indices[i]]

        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    max_count = 0
    max_index = None
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            max_index = key
    return max_index
