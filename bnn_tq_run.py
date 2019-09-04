import tensorflow as tf
import numpy as np
from hyper_dynamics import HyperDynamics
from bnn_def import BayesNN, load_MNIST
from functools import partial
import time, os, sys, imp, math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_round', type=int, default=150)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--M', type=int, default=10)
parser.add_argument('--hyperType', type=str, default='SGHMC-1') # LD, SGHMC-1, SGHMC-2
parser.add_argument('--dnType', type=str, default='Blob') # LD, Blob
parser.add_argument('--optType', type=str, default='gd') # adag, gd, sgd
parser.add_argument('--bwType', type=str, default='med')
parser.add_argument('--stepsize', type=float, default=5e-5)
parser.add_argument('--invSigma', type=float, default=1)
parser.add_argument('--diffuC', type=float, default=1)
parser.add_argument('--optRemem', type=float, default=0.99)
parser.add_argument('--optFudge', type=float, default=1e-6)
parser.add_argument('--floatName', type=str, default='None')

args = parser.parse_args()

# LD + LD = LD
# LD + Blob = Blob
# SGHMC-1 + LD = SGHMC
# SGHMC-1 + Blob
# SGHMC-2 + LD? No use
# SGHMC-2 + Blob

# Blob: LD + Blob
# SGHMC: SGHMC-1 + LD
# pSGHMC-det: SGHMC-1 + Blob
# pSGHMC-fGH: SGHMC-2 + Blob

class HP(object):
    n_repeat = 1; n_round = args.n_round; n_iter = args.n_iter
    n_hidden1 = 100; n_hidden2 = 10; n_output = 10; M = args.M
    batchsize = 500
    n_drop = 5

hp = HP()
src = imp.load_source('src', 'hyper_dynamics.py')

pm = src.HDynamicsParams(
    args.hyperType, args.dnType, args.optType, args.bwType, args.stepsize,
    dtFile = 'mnist/',
    invSigma = args.invSigma, diffuC = args.diffuC,
    optRemem=args.optRemem,
    optFudge=args.optFudge,
    # sghmc-1 sghmc-2
    optExpo=0.8
    # dnType = 'SVGD'; dnNormalize=False; accType='wnag'; accRemem=4.0; optType='sgd'; optExpo=0.8; stepsize=5e-4; bwType='med'
)

def merge_dicts(*dicts):
    return {k:v for d in dicts for k, v in d.items()}

def vars_stat(obj):
    return {k:getattr(obj, k) for k in dir(obj) if not k.startswith('_')}

if __name__ == '__main__':
    print('settings file loaded')

    np.random.seed(1)
    dataFile = './data/' + pm.dtFile
    data = load_MNIST(dataFile, batchsize=hp.batchsize)

    print('Dataset "{}" loaded.'.format(dataFile))
    print('featsize: {:d}, trainsize: {:d}, testsize: {:d}'.format(data.featsize, data.trainsize, data.testsize))
    model = BayesNN(featsize = data.featsize, batchsize=hp.batchsize, M = hp.M, n_hidden1 = hp.n_hidden1, n_hidden2 = hp.n_hidden2, n_output = hp.n_output) ##
    op_samples, dninfo = HyperDynamics(pm).evolve(model.latvar, get_logp = partial(model.get_logp, fullsize=data.trainsize))
    
    T_acc = np.zeros([hp.n_repeat, hp.n_round])
    T_llh = np.zeros([hp.n_repeat, hp.n_round])
    L_time = np.zeros([hp.n_repeat])
    for i in range(hp.n_repeat):
        if i != 0: data.reset()
        print('Repeat-trial {:d}:'.format(i))
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for j in range(hp.n_round):
                t0 = time.time()
                for k in range(hp.n_iter):
                    sess.run(op_samples, dict(zip((model.X_train, model.Y_train), data.get_batch())))
                L_time[i] += time.time() - t0
                T_acc[i,j], y_preds = sess.run([model.acc, model.y_preds], {model.X_test: data.X_test, model.Y_test: data.Y_test})
                #print(y_preds[:10])
                #print(data.Y_test[:10])
                T_acc[i,j] = np.mean(y_preds==data.Y_test)

                print('iteration {:5d}: acc {:.4f}, time {:.2f}'.format((j+1)*hp.n_iter, T_acc[i,j], L_time[i]))

    L_acc = np.max(T_acc, axis=-1)
    time_mean = np.mean(L_time)
    print('Summary: maximum_acc {:.4f}, time {:.2f}'.format(L_acc[0], time_mean))
    
    resDir = './bnn_res_mnist/'
    if not os.path.isdir(resDir): os.makedirs(resDir)
    resFile_root = resDir + args.floatName + '_' + '_'.join([args.hyperType, args.dnType, \
        args.optType, args.bwType, str(args.M)])
    
    appd = -1
    while True:
        appd += 1; resFile = resFile_root + '_{:d}.npz'.format(appd)
        if not os.path.exists(resFile): break
    print('Writing results to file "{}"'.format(resFile))
    
    np.savez(resFile,
            T_acc = T_acc, L_time = L_time,
            time_mean = time_mean,
            **merge_dicts(vars_stat(hp), vars_stat(pm)))
    
