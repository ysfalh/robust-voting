from tests.compare import run_plot
import multiprocessing as mp

SEED = 1
# TODO: make each attempt with a different seed chosen from a predefined list of seeds
# TODO: make n_subattempts for each generated original_preferences
N_PROC = mp.cpu_count()
# N_PROC = 2
DEFAULTS = {
    'n_attempts': 5, 'n_voters': 128, 'n_extreme': 0, 'n_alternatives': 256, 'density': 0.05,
    'noise': 0., 'p_byzantine': 0.33, 'byz_density': 1., 'byz_strat': 'anti', 'voting_resilience': 1.,
    'transformation_name': "min-max", 'pair_perc': 1., 'regularize': False
}

# run_plot(density=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], seed=SEED, dic=DEFAULTS)
# run_plot(pair_perc=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], seed=SEED, dic=DEFAULTS)
# run_plot(noise=[0.01, 0.1, 0.2, 0.3], seed=SEED, dic=DEFAULTS)
# run_plot(noise=[0., 0.05, 0.1, 0.2], seed=SEED, dic=DEFAULTS)
# run_plot(transformation_name=["min-max", "standardization", "median-quartile", "adversarial-0.5"], seed=SEED, dic=DEFAULTS)
run_plot(p_byzantine=[0., 0.1, 0.2, 0.3, 0.4, 0.51, 0.6, 0.7, 0.8, 0.9, 1.], seed=SEED, n_proc=N_PROC, dic=DEFAULTS)
# run_plot(voting_resilience=[0., 0.1, 1., 10., 100.], seed=SEED, dic=DEFAULTS)
# run_plot(byz_strat=['random', 'ortho', 'anti'], seed=SEED, dic=DEFAULTS)
