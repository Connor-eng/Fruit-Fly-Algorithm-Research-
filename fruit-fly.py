import numpy as np
import pandas as pd
import random
import math
from collections import defaultdict

if __name__ == '__main__':
    df_train = pd.read_csv('mnist_train_1000.csv')

    # Converting training data to NumPy array
    df_train = df_train.to_numpy()
    train_img = df_train[:, 1:].astype(int)
    train_label = df_train[:, 0].astype(int)

    # normalization of the features
    train_img = train_img / 255

    # All Variables used in this algorithm
    hashsize = 40*32 # size of the hashcodes will be 40*k (k = 32)
    p = 0.10 # 10% of the feature points will be taken for each index of the hashcode
    wta = 0.10 # top 10% of the highest firing indices will be winner take all/WTA
    zeroIndicesSize = math.ceil(0.90 * hashsize)  # Rest 90% indices should be set to zero. zeroIndicesSize indicates how many will be set to zero
    powers_of_two = 1 << np.arange(hashsize - 1, -1, step=-1)  # creates a list of 2's powers{.. ,16, 8, 4, 2, 1}

    # create the matrix M of shape (m x d) where m is size of the projections
    # and d is the size of the feature vector. M(i, j) = 1 if projection vector
    # m_i connects to x_j
    # NOTE: We will be taking 40*k hash functions. where k can be {2,4,8,16,32}
    M = np.zeros((hashsize, train_img.shape[1]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = random.uniform(0, 1)
            if val < p:
                M[i][j] = 1

    # Make a list of bins for Hashing
    bins = defaultdict(list)  # we will use built-in dictionary for now, we can implement our own if needed

    # for each input/image, calculate the hashcode using M
    # step 1: find y = Mx
    # step 2: set top 10% values to  yi and the rest to zero
    # put the input in the hashed bin after verifying they are actually similar
    for x in train_img:
        y = np.dot(M, x.T)
        zeroIndices = np.argsort(y)[: zeroIndicesSize]
        oneIndices = np.argsort(y)[zeroIndicesSize:]
        y[oneIndices] = 1
        y[zeroIndices] = 0

        # y is our binary hashcode. we want to convert this to a decimal value which we can use to index
        hashCode = np.dot(y, powers_of_two.T)

        # now add x to appropriate bin
        bins[hashCode].append(x)

        ## NOTE: WE STILL NEED TO CHECK THE SIMILARITIES BETWEEN EACH X INSIDE THE BIN's List

    