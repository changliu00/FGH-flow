class Args(object):
    dtName = 'icml'; dtFilename = './data/icml.txt'; dtVocname = './data/icml.voc'
    alpha=1e-1; beta=1e-1; sigma=1.
    K=30; batchsize=100; n_gsamp=50
    perpType = 'para'; n_window = None
    M=20
    # M=100
    # M=50
    # M=10
    # M=1
    # M=5
    n_iter=20; n_round=50
    # n_iter=20; n_round=25
    # n_iter=100; n_round=100

args = Args()
import imp
typesrc = imp.load_source('typesrc', 'hyper_dynamics.py')

pm = typesrc.HDynamicsParams('SGHMC-1', 'LD', 'gd', 'med', 1e-3,
        invSigma = 3e2, diffuC = 1e-1)
        # invSigma = 1e2, diffuC = 1e-1)
        # optExpo = .55, optIter0 = 1000)

