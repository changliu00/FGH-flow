import tensorflow as tf
import numpy as np
import sys, os

'''
    Sample code to reproduce our results for the Bayesian neural network example.
    Our settings are almost the same as Hernandez-Lobato and Adams (ICML15) https://jmhldotorg.files.wordpress.com/2015/05/pbp-icml2015.pdf
    Our implementation is also based on their Python code.
    
    p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
    p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
    p(\gamma) = Gamma(\gamma | a0, b0)
    p(\lambda) = Gamma(\lambda | a0, b0)
    
    The posterior distribution is as follows:
    p(W, \gamma, \lambda) = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda) 
    To avoid negative values of \gamma and \lambda, we update loggamma and loglambda instead.
    
    Copyright (c) 2016,  Qiang Liu & Dilin Wang
    All rights reserved.
'''

class BayesNN:
    '''
        We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.
        
        Input
            -- X_train: training dataset, features
            -- y_train: training labels
            -- batchsize: sub-sampling batch size
            -- max_iter: maximum iterations for the training procedure
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''
    _varscope_pfx = 'BayesNN_default_variable_scope_'
    _varscope_num = 0
    def __init__(self, featsize, M, batchsize=100, n_hidden1=50, n_hidden2=20, n_output=10, a0=1., b0=10., var_scope=None, reuse=None, fltype=tf.float64, Y_std=1.):
        # b0 is the scale; param in tf.random_gamma is inverse-scale
        if var_scope is None:
            var_scope = BayesNN._varscope_pfx + str(BayesNN._varscope_num)
            BayesNN._varscope_num += 1
        self.a0 = a0; self.b0 = b0; self.var_scope = var_scope
        self.M = M; self.batchsize = batchsize
        self.num_vars = featsize * n_hidden1 + n_hidden1 + \
                n_hidden1 * n_hidden2 + n_hidden2 + n_output * n_hidden2 + \
                n_output + 1

        self.X_train = tf.placeholder(dtype=fltype, shape=[None, featsize])
        self.Y_train = tf.placeholder(dtype=tf.int64, shape=[None])
        with tf.variable_scope(var_scope, reuse=reuse):
            #loggamma = tf.get_variable('loggamma', initializer=tf.log(tf.random_gamma(shape=[M, 1], alpha=a0, beta=1./b0, dtype=fltype)))
            loglambda = tf.get_variable('loglambda', initializer=tf.log(tf.random_gamma(shape=[M, 1], alpha=a0, beta=1./b0, dtype=fltype)))

            w1 = tf.get_variable('w1', shape=[M, n_hidden1, featsize], dtype=fltype, initializer=tf.random_normal_initializer(stddev=1./np.sqrt(featsize+1.)))
            b1 = tf.get_variable('b1', shape=[M, n_hidden1], dtype=fltype, initializer=tf.zeros_initializer())
            w2 = tf.get_variable('w2', shape=[M, n_output, n_hidden1], dtype=fltype, initializer=tf.random_normal_initializer(stddev=1./np.sqrt(n_hidden1+1.)))
            b2 = tf.get_variable('b2', shape=[M, n_output], dtype=fltype, initializer=tf.zeros_initializer())

            y_mean = self._get_logits(w1, b1, w2, b2, self.X_train) # X_train is fed with Unnormalized X
            #self.init_loggamma = loggamma.assign(-tf.log(tf.reduce_mean((y_mean - self.Y_train)**2, axis=1, keepdims=True))) # Y_train is fed with Unnormalized Y
            ####
        self.latvar = [w1, b1, w2, b2, loglambda]

        self.X_test = tf.placeholder(dtype=fltype, shape=[None, featsize])
        self.Y_test = tf.placeholder(dtype=tf.int64, shape=[None])
        y_logits = self._get_logits(w1, b1, w2, b2, self.X_test)
        self.y_logits = y_logits
        self.y_preds = tf.argmax(tf.reduce_mean(tf.nn.softmax(y_logits, axis=1), axis=0), axis=0) 
        self.acc = tf.reduce_sum(tf.cast(self.y_preds==self.Y_test, tf.int64))

    def _get_logits(self, w1, b1, w2, b2, X):
        X_hidden1 = tf.nn.sigmoid( tf.tensordot(w1, X, axes=[[2],[1]]) + tf.expand_dims(b1, 2))
        #X_hidden2 = tf.nn.relu( tf.matmul(w2, X_hidden1) + tf.expand_dims(b2, 2))
        X_output = tf.matmul(w2, X_hidden1) + tf.expand_dims(b2,2)

        return X_output

    def get_logp(self, w1, b1, w2, b2, loglambda, fullsize):
        # Y_train: batch_size
        # y_logits: M-by-n_output-by-batch_size -> batch_size-by-M-by-n_output
        # return: M
        y_logits = tf.transpose(self._get_logits(w1, b1, w2, b2, self.X_train), [2, 0, 1])
        loglambda = tf.squeeze(loglambda)
        mean_log_lik_data = -tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
            labels=tf.tile(tf.expand_dims(self.Y_train, 1), [1,self.M]), logits=y_logits), axis=0)

        log_prior_w = .5 * self.num_vars * (loglambda - np.log(2*np.pi)) - \
                .5*tf.exp(loglambda) * (tf.reduce_sum(w1**2, axis=[1,2]) + tf.reduce_sum(b1**2, axis=1) + \
                tf.reduce_sum(w2**2, axis=[1,2]) + tf.reduce_sum(b2**2, axis=1)) + \
                self.a0*loglambda - tf.exp(loglambda)/self.b0
        return fullsize * mean_log_lik_data + log_prior_w

class load_MNIST():

    def __init__(self, data_path, batchsize, tobinary=False):
        # We first define a download function, supporting both Python 2 and 3.
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        filename = data_path + 'MNIST'
        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

        # We then define functions for loading MNIST images and labels.
        # For convenience, they also download the requested files if needed.
        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the inputs in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            # The inputs are vectors now, we reshape them to monochrome 2D images,
            # following the shape convention: (examples, channels, rows, columns)
            #data = data.reshape(-1, 1, 28, 28)
            data = data.reshape(-1, 784)
            # The inputs come as bytes, we convert them to float32 in range [0,1].
            # (Actually to range [0, 255/256], for compatibility to the version
            # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
            return data / np.float32(256)

        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the labels in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            # The labels are vectors of integers now, that's exactly what we want.
            return data

        # We can now download and read the training and test set images and labels.
        X_train = load_mnist_images('train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
        self.X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
        self.Y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        self.X_train, self.X_dev = X_train[:-10000], X_train[-10000:]
        self.Y_train, self.Y_dev = y_train[:-10000], y_train[-10000:]

        self.featsize = self.X_train.shape[1]
        self.allsize = self.X_train.shape[0] + self.X_dev.shape[0] + self.X_test.shape[0]
        self.testsize = self.X_test.shape[0]
        self.devsize = self.X_dev.shape[0]
        self.trainsize = self.X_train.shape[0]
        self.batchsize = batchsize
        self._nIter = 0

    def reset(self):
        pass

    def get_batch(self, nIter=None):
        if nIter is not None: self._nIter = nIter
        batch = [i % self.trainsize for i in range(self._nIter * self.batchsize, (self._nIter+1) * self.batchsize)]
        self._nIter += 1
        return self.X_train[batch, :], self.Y_train[batch]

    def get_batch_for_init_loggamma(self):
        ridx = np.random.choice(self.trainsize, np.min([self.trainsize, 1000]), replace=False)
        return self.X_train[ridx, :], self.Y_train[ridx]
