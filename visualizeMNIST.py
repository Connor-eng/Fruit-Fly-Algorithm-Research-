import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df_train = pd.read_csv('mnist_train_1000.csv')

    # Converting training data to NumPy array
    df_train = df_train.to_numpy()
    train_img = df_train[:, 1:].astype(int)
    train_label = df_train[:, 0].astype(int)

    # normalization of the features
    #train_img = train_img / 255

    # visualize data
    fig = plt.figure( )
    for i in range(9):
        plt.subplot(3 , 3 , i + 1 )
        plt.imshow(np.reshape(train_img[ i ] , (28 , 28)))
        plt.title(train_label[i] )
        plt.axis('off')
    plt.tight_layout()
    plt.show()