import numpy as np
import pandas as pd


if __name__ == '__main__':
    df_train = pd.read_csv('mnist_train_1000.csv')

    # Converting training data to NumPy array
    df_train = df_train.to_numpy()
    train_img = df_train[:, 1:].astype(int)
    train_label = df_train[:, 0].astype(int)

    # normalization of the features
    train_img = train_img / 255

    # create the matrix M of shape (m x d) where m is size of the projections
    # and d is the size of the feature vector. M(i, j) = 1 if projection vector
    # m_i connects to x_j


    # Make a list of bins for Hashing

    # for each input/image, calculate the hashcode using M
    # step 1: find y = Mx
    # step 2: set top 10% values to 1 and the rest to zero
    # put the input in the hashed bin after verifying they are actually similar