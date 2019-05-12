from __future__ import division, print_function
import numpy as np
import tensorflow as tf
from six.moves import zip, map
from dynamics import Dynamics, DynamicsParams

class HDynamicsParams(DynamicsParams):
    def __init__(self, hyperType, dnType, optType, bwType, stepsize, **kwds):
        DynamicsParams.__init__(self, dnType, 'wgd', optType, bwType, stepsize, **kwds)
        self.hyperType = hyperType
        if self.bwType == 'he': self.bwSubType = 'h'
        if self.dnType == 'SVGD': self.dnNormalize = True

class HyperDynamicsInfo(object):
    def __init__(self, L_samples, L_grad_logp, var_scope, **kwds):
        self.L_samples = L_samples
        self.L_grad_logp = L_grad_logp
        self.var_scope = var_scope
        self.__dict__.update(kwds)

class HyperDynamics():
    _defl_var_scope_pfx = 'HyperDynamics_default_variable_scope_'
    _defl_var_scope_num = 0

    def __init__(self, pm): self.pm = pm

    def evolve(self, L_samples, get_logp = None, L_grad_logp = None, var_scope = None, reuse = None):
        pm = self.pm
        if var_scope is None:
            var_scope = Dynamics._defl_var_scope_pfx + str(Dynamics._defl_var_scope_num)
            Dynamics._defl_var_scope_num += 1

        if self.pm.hyperType == 'LD':
            return Dynamics(pm).evolve(L_samples, get_logp, L_grad_logp, None, var_scope, reuse)

        else:
            if type(L_samples) != list: L_samples = [L_samples]
            M = L_samples[0].get_shape()[0]
            if not all([samples.get_shape()[0] == M for samples in L_samples]): raise ValueError('Sample sizes of all variables are not the same!')
            fltype = L_samples[0].initialized_value().dtype
            if get_logp is not None:
                logp = get_logp(*L_samples)
                L_grad_logp = tf.gradients(logp, L_samples, stop_gradients=L_samples)
            elif L_grad_logp is not None:
                if type(L_grad_logp) != list: L_grad_logp = [L_grad_logp]
            else:
                raise ValueError('Exactly one of "get_logp" or "L_grad_logp" should be passed.')

            def get_name(var):
                name = var.name
                return name[(name.rfind('/')+1) : name.rfind(':')]
            L_invSigma = [tf.cast(v, fltype) for v in pm.invSigma] if type(pm.invSigma) is list else [tf.cast(pm.invSigma, fltype)] * len(L_samples)
            with tf.variable_scope(var_scope, reuse=reuse):
                L_momentum = [tf.get_variable('momentum_' + get_name(samples), initializer = tf.random_normal(shape=samples.initialized_value().get_shape(), dtype=fltype) / tf.sqrt(invSigma)) for samples, invSigma in zip(L_samples, L_invSigma)]
            L_grad_logp_mmt = [- tf.multiply(invSigma, momentum) for invSigma, momentum in zip(L_invSigma, L_momentum)]

            if self.pm.hyperType == 'HMC':
                with tf.control_dependencies(L_grad_logp):
                    return [tf.group(*(
                        [samples.assign_add(-pm.stepsize * grad_logp_mmt) for samples, grad_logp_mmt in zip(L_samples, L_grad_logp_mmt)] +
                        [momentum.assign_add(pm.stepsize * grad_logp) for momentum, grad_logp in zip(L_momentum, L_grad_logp)])),
                    HyperDynamicsInfo(L_samples, L_grad_logp, var_scope, L_momentum=L_momentum, L_grad_logp_mmt=L_grad_logp_mmt, L_invSigma=L_invSigma, stepsize=pm.stepsize)]

            elif self.pm.hyperType == 'SGHMC-1':
                L_diffuC = [tf.cast(v, fltype) for v in pm.diffuC] if type(pm.diffuC) is list else [tf.cast(pm.diffuC, fltype)] * len(L_samples)
                ##
                # with tf.control_dependencies(L_grad_logp):
                #     return [tf.group(*(
                #         [samples.assign_add(-pm.stepsize * grad_logp_mmt) for samples, grad_logp_mmt in zip(L_samples, L_grad_logp_mmt)] +
                #         [momentum.assign_add(pm.stepsize * grad_logp + tf.multiply(pm.stepsize*diffuC, grad_logp_mmt) + tf.multiply(tf.sqrt(2*pm.stepsize*diffuC), tf.random_normal(shape=momentum.initialized_value().get_shape(), dtype=fltype))) for momentum, grad_logp, diffuC, grad_logp_mmt in zip(L_momentum, L_grad_logp, L_diffuC, L_grad_logp_mmt)])),
                #     HyperDynamicsInfo(L_samples, L_grad_logp, var_scope, L_momentum=L_momentum, L_grad_logp_mmt=L_grad_logp_mmt, L_invSigma=L_invSigma, L_diffuC=L_diffuC, stepsize=pm.stepsize)]
                ##
                mmt_op, mmtinfo = Dynamics(pm).evolve(L_momentum, None, L_grad_logp_mmt, None, var_scope, reuse)
                with tf.control_dependencies(L_grad_logp):
                    return [tf.group(*(
                        [samples.assign_add(-mmtinfo.stepsize * grad_logp_mmt) for samples, grad_logp_mmt in zip(L_samples, L_grad_logp_mmt)] +
                        [momentum.assign_add(mmtinfo.stepsize * grad_logp + tf.multiply(diffuC, increm_mmt)) for momentum, grad_logp, diffuC, increm_mmt in zip(L_momentum, L_grad_logp, L_diffuC, mmtinfo.L_increm)])),
                    HyperDynamicsInfo(L_samples, L_grad_logp, var_scope, L_momentum=L_momentum, L_grad_logp_mmt=L_grad_logp_mmt, mmtinfo=mmtinfo, mmt_op=mmt_op, L_invSigma=L_invSigma, L_diffuC=L_diffuC)]
                ##

            elif self.pm.hyperType == 'SGHMC-2':
                L_diffuC = [tf.cast(v, fltype) for v in pm.diffuC] if type(pm.diffuC) is list else [tf.cast(pm.diffuC, fltype)] * len(L_samples)
                mmt_op, mmtinfo = Dynamics(pm).evolve(L_momentum, None, L_grad_logp_mmt, None, var_scope, reuse)
                smp_op, smpinfo = Dynamics(pm).evolve(L_samples, None, L_grad_logp)
                with tf.control_dependencies(smpinfo.L_increm):
                    return [tf.group(*(
                        [samples.assign_add(-increm_mmt) for samples, increm_mmt in zip(L_samples, mmtinfo.L_increm)] +
                        [momentum.assign_add(increm_smp + tf.multiply(diffuC, increm_mmt)) for momentum, increm_smp, diffuC, increm_mmt in zip(L_momentum, smpinfo.L_increm, L_diffuC, mmtinfo.L_increm)])),
                    HyperDynamicsInfo(L_samples, L_grad_logp, var_scope, L_momentum=L_momentum, L_grad_logp_mmt=L_grad_logp_mmt, mmtinfo=mmtinfo, smpinfo=smpinfo, mmt_op=mmt_op, smp_op=smp_op, L_invSigma=L_invSigma, L_diffuC=L_diffuC)]

            else:
                raise ValueError('unknown "hyperType": "{}"!'.format(self.pm.hyperType))

