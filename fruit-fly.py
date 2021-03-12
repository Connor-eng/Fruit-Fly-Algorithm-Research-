import numpy as np
import pandas as pd
import random
import math
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# def jaccard_similarity(list1, list2):
#     intersection = len(list(set(list1).intersection(list2)))
#     union = (len(list1) + len(list2)) - intersection
#     return float(intersection) / union

def getHashCode(M, x, zeroIndicesSize, powers_of_two):
    y = np.dot(M, x.T)
    zeroIndices = np.argsort(y)[: zeroIndicesSize]
    oneIndices = np.argsort(y)[zeroIndicesSize:]
    y[oneIndices] = 1
    y[zeroIndices] = 0

    # y is our binary hashcode. we want to convert this to a decimal value which we can use to index
    hashCode = np.dot(y, powers_of_two.T)
    return hashCode

# given a input X of shape (n x d) and a target of shape(1 x d), return indices of all neighbors of the target
# result will be of shape (m x d) ..here n = number of inputs, d = number of dimensions, m = number of neighbors
def findNeighbors(X, target):
    # normalize X and Target
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # All Variables used in this algorithm
    similarity = 0.80 # we will take data that matches atleast 80%
    hashsize = 40 * 32  # size of the hashcodes will be 40*k (k = 32)
    p = 0.10  # 10% of the feature points will be taken for each index of the hashcode
    #top 10% of the highest firing indices will be winner take all/WTA
    zeroIndicesSize = math.ceil(0.90 * hashsize)  # Rest 90% indices should be set to zero. zeroIndicesSize indicates how many will be set to zero
    powers_of_two = 1 << np.arange(hashsize - 1, -1, step=-1)  # creates a list of 2's powers{.. ,16, 8, 4, 2, 1}

    # create the matrix M of shape (m x d) where m is size of the projections
    # and d is the size of the feature vector. M(i, j) = 1 if projection vector m_i connects to x_j
    # NOTE: We will be taking 40*k hash functions. where k can be {2,4,8,16,32}
    M = np.zeros((hashsize, X_norm.shape[1]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = random.uniform(0, 1)
            if val < p:
                M[i][j] = 1

    # Make a list of bins for Hashing
    bins = defaultdict(list)  # we will use built-in dictionary for now, we can implement our own if needed

    # for each input, calculate the hashcode using M
    for i, x in enumerate(X_norm):
        hashCode = getHashCode(M, x, zeroIndicesSize, powers_of_two)

        # now add indices of the input to appropriate bin
        bins[hashCode].append(i)
    print(len(bins))
    ## NOTE: WE STILL NEED TO CHECK THE SIMILARITIES BETWEEN EACH X INSIDE THE BIN's List with the target

    #get the idices from the bin where target is hashed
    return bins[getHashCode(M, target, zeroIndicesSize, powers_of_two)]

    # visualize the elements in the first bin
    # for _, val in bins.items():
    #     for i in range(min(len(val), 9)):
    #         plt.subplot(3, 3, i + 1)
    #         plt.imshow(np.reshape(val[i], (28, 28)))
    #         plt.axis('off')
    #     plt.tight_layout()
    #     plt.show()
    #     break

if __name__ == '__main__':

    df_mnist = pd.read_csv('mnist_train_1000.csv')
    df_iris = pd.read_csv('Iris.csv')
    df_cars = pd.read_csv('auto-mpg.csv')
    df_wines = pd.read_csv('wine-data.csv')

    # Converting training data to NumPy array
    df_mnist_inmage = df_mnist.to_numpy()[:, 1:]
    df_mnist_label = df_mnist.to_numpy()[:, 0]

    df_iris_data = df_iris.to_numpy()[:, 1:-1]
    df_iris_label= df_iris.to_numpy()[:, -1]

    df_cars_data = df_cars.to_numpy()[:, : -1].astype(np.float)
    df_cars_label= df_cars.to_numpy()[:, -1]

    df_wines_data = df_wines.to_numpy()[:, 1:]
    df_wines_label = df_wines.to_numpy()[:, 0]


    print(df_iris_label[ findNeighbors(df_iris_data, df_iris_data[0, :]) ])