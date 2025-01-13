import numpy as np
import os
import copy
# L2_norm_np = np.linalg.norm(data, ord=2, axis=1)  # ord=2, L2 范数归一化

# norm_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# print(norm_array.shape)
# sum_col = np.sum(norm_array, axis=0)  # 按列加
# print('sum_col: ', sum_col)
# sum_row = np.sum(norm_array, axis=1)  # 按行加
# print('sum_row: ', sum_row)


def norm_l2_for_each_row(data_numpy):
    for i in range(data_numpy.shape[0]):
        l2_norm = 0
        # data_numpy_i = data_numpy[i]

        for j in range(data_numpy.shape[1]):
            l2_norm += data_numpy[i][j] ** 2
        l2_norm = np.sqrt(l2_norm)
        # print('L2_norm: ', L2_norm)
        # print('L2_norm_np: ', L2_norm_np[i])

        # L2_norm_str = str(L2_norm)[:6]
        # L2_norm_np_str = str(L2_norm_np[i])[:6]

        # print(str(L2_norm)[:6] == str(L2_norm_np[i])[:6])
        if l2_norm != 0.0:
            data_numpy[i] = data_numpy[i]/l2_norm

        # print(data[i])
    return data_numpy


def norm_l2_for_all_event(data_numpy):
    l2_norm = 0
    for i in range(data_numpy.shape[0]):
        for j in range(data_numpy.shape[1]):
            l2_norm += data_numpy[i][j] ** 2
    l2_norm = np.sqrt(l2_norm)
    if l2_norm != 0.0:
        data_numpy = data_numpy / l2_norm
        # destination = np.sum(data_numpy, axis=0)
        return data_numpy
    else:
        return data_numpy


if __name__ == '__main__':
    # n = 5
    # data_1d_0_1 = np.linspace(start=1, stop=10, num=n ** 2)
    # data = np.reshape(data_1d_0_1, (n, n))

    # print('original', data)
    # data = norm_l2_for_each_row(data)

    # print('normalized', data)

    norm_array_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    data_norm = norm_l2_for_all_event(norm_array_1)


