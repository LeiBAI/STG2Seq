import numpy as np
import scipy.io
import math

def generate_graph_with_map(dir):
    graph = np.zeros((35, 35))
    _table = scipy.io.loadmat(dir)
    _region_data = _table['L'] - 1
    for i in range(_region_data.shape[0] - 1):
        for j in range(_region_data.shape[1] - 1):
            if _region_data[i, j] != _region_data[i, j+1]:
                graph[_region_data[i, j], _region_data[i, j+1]] = 1
                graph[_region_data[i, j + 1], _region_data[i, j]] = 1
            elif _region_data[i, j] != _region_data[i+1, j+1]:
                graph[_region_data[i, j], _region_data[i+1, j+1]] = 1
                graph[_region_data[i+1, j+1], _region_data[i, j]] = 1
            elif _region_data[i, j] != _region_data[i+1, j]:
                graph[_region_data[i, j], _region_data[i+1, j]] = 1
                graph[_region_data[i+1, j], _region_data[i, j]] = 1
    return graph

def generate_graph_with_data(data, length, threshold=0.05):
    #data shape is [sample_nums, node_nums, dims] or [sample_nums, node_nums]
    if len(data.shape) == 2:
        dim = 1
        data = np.expand_dims(data, axis=-1)
    if len(data.shape) == 3:
        dim = data.shape[2]
    else:
        print('Wrong data format! Shape of data is:', data.shape)
        exit(0)
    node_num = len(data[0, :, 0])
    adj_mx = np.zeros((node_num, node_num))
    demand_zero = np.zeros((length, dim))
    for i in range(node_num):
        node_i = data[-length:, i, :]
        adj_mx[i, i] = 1
        if np.array_equal(node_i, demand_zero):
            continue
        else:
            for j in range(i + 1, node_num):
                node_j = data[-length:, j, :]
                distance = math.exp(-(np.abs((node_j - node_i)).sum() / length*dim))
                if distance > threshold:
                    adj_mx[i, j] = 1
                    adj_mx[j, i] = 1
    sparsity = adj_mx.sum() / (node_num * node_num)
    print("Sparsity of the adjacent matrix is: ", sparsity)
    #print(adj_mx)
    return adj_mx

def generate_graph_with_poi(poi):
    pass


if __name__ == '__main__':
    graph = generate_graph_with_map("../data/region.mat")
    np.savetxt("../data/adj_mx.csv", graph, fmt='%d', delimiter=',')
