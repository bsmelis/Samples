
from __future__ import print_function, division
from builtins import range,input
from sklearn.utils import shuffle


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class AutoEncoder(object):
    def __init__(self,D,M):
        self.M = M
        self.build(D,M)

    def build(self, D, M):
        self.W = tf.Variable(tf.random_normal(shape=(D,M)))
        self.bh = tf.Variable(tf.zeros(M))
        self.bo = tf.Variable(tf.zeros(D))

        self.X_in = tf.placeholder(tf.float32, shape=(None,D))
        self.Z = tf.nn.sigmoid(tf.matmul(self.X_in,self.W)+self.bh)
        self.X_hat = tf.nn.sigmoid(tf.matmul(self.Z,tf.transpose(self.W))+self.bo)
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X_in, logits=self.X_hat))
        self.train_op = tf.train.AdamOptimizer(1e-1).minimize(self.cost)

    def fit(self,X,sess,epochs=1, batch_sz = 100):
        self.session = sess
        N,D = X.shape
        n_batches = N//batch_sz
        costs=[]
        for i in range(epochs):
            X=shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                _,c =self.session.run((self.train_op, self.cost), feed_dict={self.X_in:batch})
                print(c)
                costs.append(c)
        plt.plot(costs)
        plt.show()
        
#################################################Test with MNIST dataset####################################
def loadMNIST():

    train = pd.read_csv('train.csv').as_matrix().astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000,1:] / 255
    Ytrain = train[:-1000,0].astype(np.int32)

    Xtest  = train[-1000:,1:] / 255
    Ytest  = train[-1000:,0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest

Xtrain, Ytrain, Xtest, Ytest = loadMNIST()
Xtrain = Xtrain.astype(np.float32)
Xtest = Xtest.astype(np.float32)
Q,D = Xtrain.shape
encod = AutoEncoder(D,300)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    encod.fit(Xtrain,sess,50)
