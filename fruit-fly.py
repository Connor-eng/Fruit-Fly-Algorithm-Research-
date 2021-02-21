import numpy as np
import pandas as pd
import random

if __name__ == '__main__':
    df_train = pd.read_csv('mnist_train_1000.csv')

    # Converting training data to NumPy array
    df_train = df_train.to_numpy()
    train_img = df_train[:, 1:].astype(int)
    train_label = df_train[:, 0].astype(int)

    # normalization of the features
    train_img = train_img / 255

    # All Variables used in this algorithm
    k = 32 # size of the hashcodes will be 40*k
    p = 0.10 # 10% of the feature points will be taken for each index of the hashcode

    # create the matrix M of shape (m x d) where m is size of the projections
    # and d is the size of the feature vector. M(i, j) = 1 if projection vector
    # m_i connects to x_j
    # NOTE: We will be taking 40*k hash functions. where k can be {2,4,8,16,32}
    M = np.zeros((40*k, train_img.shape[1]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = random.uniform(0, 1)
            if val < p:
                M[i][j] = 1

    # Make a list of bins for Hashing

    # for each input/image, calculate the hashcode using M
    # step 1: find y = Mx
    # step 2: set top 10% values to 1 and the rest to zero
    # put the input in the hashed bin after verifying they are actually similar