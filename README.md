# [Understanding MCMC Dynamics as Flows on the Wasserstein Space](http://proceedings.mlr.press/v97/liu19j.html)

[Chang Liu][changliu] \<~~<chang-li14@mails.tsinghua.edu.cn>~~; <liuchangsmail@gmail.com>\>,
[Jingwei Zhuo][jingweizhuo], and [Jun Zhu][junzhu]. ICML 2019.

\[[Paper & Appendix](http://ml.cs.tsinghua.edu.cn/~changliu/hgf/hgf.pdf)\]
\[[Slides](http://ml.cs.tsinghua.edu.cn/~changliu/hgf/hgf_beamer.pdf)\]
\[[Poster](http://ml.cs.tsinghua.edu.cn/~changliu/hgf/hgf_poster.pdf)\]

## Introduction

The project aims to interpret a general MCMC dynamics ([Ma et al., 2015](https://papers.nips.cc/paper/5891-a-complete-recipe-for-stochastic-gradient-mcmc)).
This is [done](https://www-dimat.unipv.it/savare/Ravello2010/JKO.pdf)
for the [Langevin dynamics (LD)](https://link.springer.com/content/pdf/10.1023/A:1023562417138.pdf),
which simulates the gradient flow of the KL divergence on the Wasserstein space
thus steepest minimizes the difference from the target posterior distribution.
This does not hold for general MCMC dynamics, which only guarantee
that the target distribution is kept invariant.
In this work, we develop some mathematical concepts and reveal that a general MCMC dynamics
corresponds to the composition of the _Hamiltonian flow_ and the so-called _fiber-gradient flow_
of the KL divergence, where the former conserves the KL on the Wasserstein space of the support space
and the latter minimizes KL on each of the Wasserstein spaces
of the fibers (a set of certain subspaces) of the support space.
An MCMC dynamics specifies the geometric structures for determining the two flows,
thus its behavior can be made clear under this picture, e.g.,
the [instability](http://proceedings.mlr.press/v37/betancourt15.html "Betancourt, 2015")
of [HMC](https://arxiv.org/abs/1206.1901 "Neal, 2011")
in face of stochastic gradient as opposed to [LD](http://www.jmlr.org/papers/volume17/teh16a/teh16a.pdf)
and [SGHMC][sghmc-paper],
and the faster convergence of SGHMC over LD.
Moreover, this interpretation also facilitates particle-based variational inference methods (ParVIs)
to go beyond the current dynamics scope of LD and use more efficient dynamics.
As an example, we develop two novel ParVIs that use the SGHMC dynamics.

The repository here implements the proposed ParVIs along with existing ParVIs and MCMCs (LD and SGHMC).
The experiments demonstrate that the proposed ParVIs converge faster than existing ParVIs
due to the better efficiency of SGHMC over LD, and that they are more particle-efficient than SGHMC,
which is the advantage of ParVIs.
The implementations are built based on the Python code with [TensorFlow](https://www.tensorflow.org/)
by [Liu et al. (2019)](https://github.com/chang-ml-thu/AWGF).

## Instructions
* For the synthetic experiment:

	Directly open "synth_run.ipynb" in a jupyter notebook.

* For the Latent Dirichlet Allocation experiment:

	First run
	```bash
	python lda_build.py build_ext --inplace
	```
	to compile the [Cython](https://cython.org/) code, then run
	```bash
	python lda_run.py ./lda_sett_icml/[a specific settings file]
	```
	to conduct experiment under the specified settings.

	The ICML dataset ([download here](https://cse.buffalo.edu/~changyou/code/SGNHT.zip))
	is developed and utilized by [Ding et al. (2015)](http://papers.nips.cc/paper/5592-bayesian-sampling-using-stochastic-gradient-thermostats).

	Codes are developed based on the codes of [Patterson & Teh (2013)](http://www.stats.ox.ac.uk/~teh/sgrld.html)
	for their work "[Stochastic Gradient Riemannian Langevin Dynamics for Latent Dirichlet Allocation](https://papers.nips.cc/paper/4883-stochastic-gradient-riemannian-langevin-dynamics-on-the-probability-simplex)".

* For the Bayesian neural network experiment:

	Directly edit the file "bnn_tq_run.py" to make a setting, and run
	```bash
	python bnn_tq_run.py
	```
	to conduct experiment under the specified settings.
	The experiment setup follows that of [Chen et al. (2014)][sghmc-codes]
	in their work "[Stochastic Gradient Hamiltonian Monte Carlo][sghmc-paper]".

## Citation
```
@InProceedings{liu2019understanding_b,
  title = 	 {Understanding {MCMC} Dynamics as Flows on the {W}asserstein Space},
  author = 	 {Liu, Chang and Zhuo, Jingwei and Zhu, Jun},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {4093--4103},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/liu19j/liu19j.pdf},
  url = 	 {http://proceedings.mlr.press/v97/liu19j.html},
  organization={IMLS},
}
```

[changliu]: http://ml.cs.tsinghua.edu.cn/~changliu/index.html
[junzhu]: http://ml.cs.tsinghua.edu.cn/~jun/index.shtml
[jingweizhuo]: http://ml.cs.tsinghua.edu.cn/~jingwei/index.html
[svgd-paper]: http://papers.nips.cc/paper/6338-stein-variational-gradient-descent-a-general-purpose-bayesian-inference-algorithm
[svgd-codes]: https://github.com/DartML/Stein-Variational-Gradient-Descent
[sghmc-paper]: http://proceedings.mlr.press/v32/cheni14.html
[sghmc-codes]: https://github.com/tqchen/ML-SGHMC

