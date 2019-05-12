# Understanding MCMC Dynamics as Flows on the Wasserstein Space
## [Chang Liu](https://github.com/chang-ml-thu), Jingwei Zhuo, and Jun Zhu

## Instructions
* For the synthetic experiment:
	Directly open "synth_run.ipynb" in a jupyter notebook.

* For the Latent Dirichlet Allocation experiment:
	First run
		
		"python lda_build.py build_ext --inplace"

	to compile the Cython code, then run

		"python lda_run.py ./lda_sett_icml/[a specific settings file]"

	to conduct experiment under the specified settings.
	The ICML dataset can be downloaded from
	
		https://cse.buffalo.edu/ ̃changyou/code/SGNHT.zip

	Codes are developed based on the codes of "Stochastic Gradient Riemannian Langevin Dynamics for Latent Dirichlet Allocation" (Patterson and Teh, 2013).

* For the Bayesian neural network experiment:
	Directly edit the file "bnn_tq_run.py" to make a setting, and run

		"python bnn_tq_run.py"
	
	to conduct experiment under the specified settings.
	Experiment setup follows the one of "Stochastic Gradient Hamiltonian Monte Carlo" (Chen et al., 2014)

