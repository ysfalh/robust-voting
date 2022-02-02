from lib2to3.pgen2.pgen import DFAState
from tests.compare import run_plot, multiple_experiments

SEED = 500    # TODO: make each attempt with a different seed chosen from a predefined list of seeds
DEFAULTS = {
    'n_attempts': 1, 'n_voters': 61, 'n_extreme': 0, 'n_alternatives': 100, 'density': 0.1,
    'noise': 0.05, 'p_byzantine': 0, 'byz_density': 0, 'byz_strat':'random', 'voting_resilience': 1.,
    'transformation_name': "median-quartile", 'pair_perc': 1, 'sm3':1, 'sm4':1
}

# run_plot(density=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], seed=SEED, dic=DEFAULTS)
# run_plot(pair_perc=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], seed=SEED, dic=DEFAULTS)
# run_plot(noise=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], seed=SEED, dic=DEFAULTS)
# run_plot(noise=[0., 0.05, 0.1, 0.2], seed=SEED, dic=DEFAULTS)
# run_plot(transformation_name=["min-max", "standardization", "median-quartile", "adversarial-0.5"], seed=SEED, dic=DEFAULTS)
# run_plot(p_byzantine=[0., 0.1, 0.2, 0.3, 0.4, 0.51], seed=SEED, dic=DEFAULTS)

# run_plot(n_extreme=[0, 5, 10, 15], seed=SEED, dic=DEFAULTS)
# run_plot(byz_strat=['random', 'ortho', 'anti'], seed=SEED, dic=DEFAULTS)
EXPERIMENTS = {
    'p_byzantine':[0., 0.51],
    'noise':[0., 0.06],
    'pair_perc':[0.01, 0.05, 0.1]

}

multiple_experiments(EXPERIMENTS, defaults=DEFAULTS, seed=SEED)
