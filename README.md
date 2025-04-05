# Bayesian Experimental Design Through Reinforcement Learning

Notes: This repository and its contents support a master's thesis under the INM363 module at City, University of London. As of March 2025, a paper based on this master's thesis, and as a result this repository, was presented at the [AAAI 2025 '8th Workshop on Generalization in Planning'](https://aair-lab.github.io/genplan25/). The paper is titled [Performance Comparisons of Reinforcement Learning Algorithms for Sequential Experimental Design](https://arxiv.org/abs/2503.05905).

The code presented here is an updated version of the work found in the [RL-BOED](https://github.com/csiro-mlai/RL-BOED) repository and corresponding paper [Optimizing Sequential Experimental Design with Deep Reinforcement Learning](https://arxiv.org/abs/2202.00821). This version works with the latest versions of PyTorch, NumPy, and Gymnasium (note the difference from Gym). Since [Akro](https://github.com/rlworkgroup/akro) and [Garage](https://github.com/rlworkgroup/garage/) do not support the latest versions of these libraries (at least as of June 2024), they are imported and edited in this repository as separate folders. Note that we use the [2021.03 release](https://github.com/rlworkgroup/garage/tree/release-2021.03) of Garage since it was used in [RL-BOED](https://github.com/csiro-mlai/RL-BOED).

### Requirements
- Python 3.9+ - we use Python 3.9.5
- [PyTorch (with CUDA for GPU usage)](https://pytorch.org/get-started/locally/) - we use PyTorch 2.3.0
- All other requirements listed in [**requirements.txt**](requirements.txt) - specific versions are listed

## Background

Optimal experimental design is the area dedicated to the optimal execution of experiments, with respect to some allocation of resources. Much work on optimal experimental design falls under the Bayesian setting, where we seek to reduce the uncertainty about our parameters of interest through experimentation. Conducting these experiments sequentially has recently brought about the use of reinforcement learning, where an agent is trained to navigate the design space to select the most informative designs for experimentation. However, there is still a lack of understanding about the benefits and drawbacks of using certain reinforcement learning algorithms to train these agents. In our work, we explore several reinforcement learning algorithms based on the state-of-the-art soft actor-critic method, and apply these to three Bayesian experimental design problems. We examine the amount of time needed to train these agents through each algorithm, and assess the generalisability of each agent to different but related experimental design setups. We draw insights on which algorithm generally performs best, and under what circumstances one may wish to use a particular algorithm.

## Data

Our trained agents, alongside their results at evaluation time, can be found [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EpDON-jNQRlElzC_crrOVd8BLTZeAa3YfN-BfrNehvAiCA?e=gRNk9z). Each folder beginning with 'boed_results' represents a set of 10 agents, each trained on a unique random seed. The agents are trained with a specific algorithm under a certain set of hyperparameters, and on a particular Bayesian experimental design problem. Folders with 'ces' are agents trained on the 'Constant Elasticity of Substitution' experiment, 'location' is the 'Location Finding' experiment, and 'docking' is the 'Biomolecular Docking' experiment.

## Algorithms

We explore [Randomised Ensembled Double Q-Learning (REDQ)](https://arxiv.org/abs/2101.05982), [Dropout Q-Functions for Doubly Efficient Reinforcement Learning (DroQ)](https://arxiv.org/abs/2110.02034), [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier (SBR)](https://openreview.net/forum?id=OpC-9aBBVJe), and [A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning (SUNRISE)](https://arxiv.org/abs/2007.04938). These all extend the [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905) algorithm. [Proximal Policy Optimisation (PPO)](https://arxiv.org/abs/1707.06347) and [Trust Region Policy Optimisation (TRPO)](https://arxiv.org/abs/1502.05477) are two other algorithms that can be used.

## Bayesian Experimental Design Problems

### Location Finding/Source Location

There are $K$ objects on a $d$-dimensional space, and in this experiment we need to identify their locations $\boldsymbol{\theta} = \{\theta_{i}\}_{i = 1}^{K}$. Each object emits a signal, which obeys the inverse-square law. We need to select different designs $\xi$, which are essentially locations (or points) on the $d$-dimensional space. The signal strength increases as we select $\xi$ closer to these $K$ objects, and it decays as we choose $\xi$ further away from these objects.

The total intensity at point $\xi$ is the superposition of the individual intensities for each object, $$\mu(\boldsymbol{\theta}, \xi) = b + \sum_{i = 1}^{K} \frac{\alpha}{m + ||\theta_{i} - \xi||^{2}},$$ where $\alpha$ is a constant, $b > 0$ is a constant controlling the background signal, and $m > 0$ is a constant controlling the maximum signal. The total intensity is then used in the likelihood function calculation.

For object $\theta_{i} \in \mathbb{R}^{d}$, its prior is given by $$\theta_{i} \sim \mathcal{N}_{d}(\boldsymbol{0}, I),$$ where $\boldsymbol{0}$ is the mean vector, and $I$ is the covariance matrix, an identity matrix, both with dimension $d$. For a given design $\xi$, the likelihood function is given by $$y \mid \boldsymbol{\theta}, \xi \sim \mathcal{N}(\log \mu(\boldsymbol{\theta}, \xi), \sigma^2).$$

| Parameter  | Value |
| ------------- | ------------- |
| $K$ | 2 |
| $d$ | 2 |
| $\alpha$ | 1 |
| $b$ | 0.1 |
| $m$ | 0.0001 |
| $\sigma$ | 0.5 |

### Constant Elasticity of Substitution

We have two baskets $\boldsymbol{x}, \boldsymbol{x'} \in [0, 100]^{3}$ of goods, and a human indicates their preference of the two baskets on a sliding 0-1 scale. For example, one basket could contain two chocolate bars, while the other basket contains an apple. The exact items in the baskets are indicated numerically and are not exactly real-world items, at least in our model. The CES model \citep{arrowchen} with latent variables $(\rho, \boldsymbol{\alpha}, u)$, which all characterise the human’s utility or preferences for the different items, is then used to measure the difference in utility of the baskets. The goal is to design the baskets in a way that allows for inference of the latent variable values, otherwise reducing our uncertainty about them. Both baskets are 3-tuples, meaning that we have $3 + 3 = 6$ design space dimensions $(\xi = (\boldsymbol{x}, \boldsymbol{x'}))$.

The CES model \citep{arrowchen} defines the utility $U(\boldsymbol{x})$ for a basket of goods $\boldsymbol{x}$ as, $$U(\boldsymbol{x}) = \left( \sum_{i} x_i^{\rho} \alpha_{i} \right)^{\frac{1}{\rho}},$$ where $\rho$ and $\boldsymbol{\alpha}$ are latent variables defined with the prior distributions explained below. This utility function, which is a measure of satisfaction in economic terms, is then used in the likelihood function calculation.

We use the following priors for $(\rho, \boldsymbol{\alpha}, u)$:
$$\rho \sim \text{Beta}(1, 1), \boldsymbol{\alpha} \sim \text{Dirichlet}([1, 1, 1]), u \sim \text{Log-Normal}(1, 3^{2}).$$

The likelihood function is the preference of the human on a sliding 0-1 scale, which is based on $U(\boldsymbol{x}) - U(\boldsymbol{x'})$. For a given design $\xi$, the likelihood function is given by, $$\mu_{\eta} = u \cdot (U(\boldsymbol{x}) - U(\boldsymbol{x'})), \sigma_{\eta} = \nu u \cdot (1 + \|\boldsymbol{x} - \boldsymbol{x'}\|), \eta \sim \mathcal{N}(\mu_{\eta}, \sigma_{\eta}^{2}), y = \text{clip}(s(\eta), \epsilon, 1 - \epsilon),$$
where $\nu = 0.005$, $\epsilon = 2^{-22}$, and $s(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function. Notice that the normally distributed $\eta$ is passed through the sigmoid function, bounding $\eta \in [0, 1]$. Censoring/clipping is applied, which limits the distribution by setting values above $u = 1 - \epsilon$ to be equal to $1 - \epsilon$, and values below $l = \epsilon$ to be equal to $\epsilon$.

### Biomolecular Docking

Molecular docking \citep{Meng_XuanYu2011} explores how two or more molecular structures interact with each other. When a compound and receptor bind, this is known as a `hit'. In this experiment, we need to select the most informative compounds to find the predicted binding affinity (docking score) ranges from which molecules would be picked for testing in an experiment.

The probability of outcome $y_{i}$ being a hit, given a docking score $\xi \in [-75, 0]$, is modelled as a sigmoid hit-rate model. We have $\boldsymbol{\theta} = (\textit{top}, \textit{bottom}, \textit{ee50}, \textit{slope})$, where 'top' is the maximum hit-percent, 'ee50' is the dock energy in kcal/mol at 'top'/2, 'slope' is the change in hit-percent at 'ee50' in hit-percent/(kcal/mol), and 'bottom' is the minimum hit-rate.

We use the following priors for $(\textit{top}, \textit{bottom}, \textit{ee50}, \textit{slope})$: $$\textit{top} \sim \text{Beta}(25, 75), \textit{bottom} \sim \text{Beta}(4, 96), \textit{ee50} \sim \mathcal{N}(-50, 15^{2}), \textit{slope} \sim \mathcal{N}(-0.15, 0.1^{2}).$$

The likelihood function is Bernoulli distributed and provides a binary outcome as to whether or not the docking score leads to a hit. 1 means that there is a hit, and 0 means there is no-hit. For a given design $\xi$, the likelihood function is given by, $$y_{i} \mid \boldsymbol{\theta}, \xi \sim \text{Bernoulli} \bigg(\textit{bottom} + \frac{\textit{top} - \textit{bottom}}{1 + \exp\left(-\textit{slope}(\xi_{i} - \textit{ee50})\right)}\bigg).$$

## Code

### File Structure

See the arguments for each script at the end of the code, for example `process_results.py` can be written in command line as (with relevant directories input):

``python3 process_results.py --fpaths="Documents\Training Results\boed_results_sbr_430000\source\progress.csv, Documents\Training Results\boed_results_sbr_430000\source_1\progress.csv, Documents\Training Results\boed_results_sbr_430000\source_2\progress.csv, Documents\Training Results\boed_results_sbr_430000\source_3\progress.csv, Documents\Training Results\boed_results_sbr_430000\source_4\progress.csv" --dest="Documents\Training Results\sbr430000_results.npz"``

- `Adaptive_{env}_{algo}.py`: File to initiate the training loop for the respective environment/experimental design problem {env} and algorithm {algo}; {env - Source}: Location Finding, {env - CES}: Constant Elasticity of Substitution, {env - Docking}: Biomolecular Docking.
- `process_results.py`: Produces training datasets with the training performance from several random seeds, for a particular environment and algorithm.
- `plot_results.py`: Allows training performance results to be plot, using the data files produced by `process_results.py`.
- `select_policy_env.py`: Evaluate/test a trained policy on a particular experimental design problem (the exact one it was trained on, or one with slightly different experimental parameters).

If you have access to a high-performance computer through SLURM, we recommend making use of the Bash scripts in [**useful-bash-scripts**](/useful-bash-scripts).

### Training the Agents

The experiments are quite expensive to run on a standard PC, and so we advise looking at using a high-performance computer. We use SLURM to run our experiments on NVIDIA A100 80GB PCIe and NVIDIA A100 40GB PCIe GPUs with 4 CPU cores and 40GB of RAM. An Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz was used.

We note that DroQ can be run through REDQ, by setting 'ens-size' and 'M' to be equal to each other (say 2), 'layer-norm = True', and 'dropout > 0' for the dropout probability (which should be larger than 0 for DroQ).

We can choose a reward function based on the sequential prior contrastive estimation (sPCE), which is a lower bound on the expected information gain. This is the standard due to it being bounded by \log (n-contr-samples + 1), unlike the upper bound sequential nested Monte Carlo (sNMC). sPCE can be selected by setting 'bound-type = lower', and sNMC can be selected using 'bound-type = upper'. Rewards are dense here, meaning that the incremental sPCE/sNMC is provided to the agent during training in each experiment $t$. Setting 'bound-type = terminal' returns to the sparse reward setting, where the agent only receives a reward at the end of experimentation, which is the full sPCE.

After deciding on an environment and algorithm to use, either run the relevant Python file on your IDE, or use Bash/command line to run the Python file with your chosen (environment and algorithm specific) values to parse (more values can be parsed, see the Python files):

``python3 Adaptive_Source_REDQ.py --n-parallel=100 --n-contr-samples=100000 --n-rl-itr=20001 --log-dir="boed_results_discount_0.99/source"  --bound-type=lower --id=$1 --budget=30 --discount=0.99 --buffer-capacity=10000000 --tau=0.001 --pi-lr=0.0001 --qf-lr=0.0003 --M=2 --ens-size=2 --lstm-q-function=False --layer-norm=False --dropout=0``

Your choice of random seeds to experiment with can be entered near the beginning of the Python file, replacing the 'seeds' variable with your chosen seeds to explore. The code understands which seed to use through the parsed 'id' value in Bash, if 'id = 1', then the first seed would be used for training, and so on. The Bash scripts we use loop over each of the 10 seeds in the list of the Python file, and run 10 jobs on SLURM, training an agent using a seed from the list.

The saved agent and its training results will be available in the directory named in 'log-dir'.

### Evaluating the Agents

Once an agent has been trained, it can be evaluated on the same environment it was trained on. The environment specific parameters, such as the number of locations $K$ in the Location Finding problem, can be edited here. This can be a method of testing generalisability to alternative scenarios. Testing the agent on both sPCE and sNMC can be done here too. Results of the final sPCE/sNMC values are located in the file (ideally .txt or .log) in 'dest'.

``python3 select_policy_env.py --src="boed_results_sbr_430000/source_9/itr_20000.pkl" --dest="boed_results_sbr_430000/source_9/evaluation_lower.log" --edit_type="w" --seq_length=30 --bound_type=lower --n_contrastive_samples=1000000 --n_parallel=250 --n_samples=2000 --seed=1 --env="source" --source_d=2 --source_k=5 --source_b=0.1 --source_m=0.0001 --source_obs_sd=0.5 --ces_d=6 --ces_obs_sd=0.005 --docking_d=1``
