from tests.compare import run_plot

SEED = 1    # TODO: make each attempt with a different seed chosen from a predefined list of seeds
DEFAULTS = {
    'n_attempts': 10, 'n_voters': 25, 'n_extreme': 0, 'n_alternatives': 100, 'density': 0.1,
    'noise': 0., 'p_byzantine': 0.33, 'byz_density': 0.33, 'voting_resilience': 1.,
    'transformation_name': "min-max", 'pair_perc': 0.8, 'regularize': False
}

run_plot(density=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], seed=SEED, dic=DEFAULTS)
# run_plot(pair_perc=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], seed=SEED, dic=DEFAULTS)
# run_plot(noise=[0.01, 0.1, 0.2, 0.3], seed=SEED, dic=DEFAULTS)
# run_plot(noise=[0., 0.05, 0.1, 0.2], seed=SEED, dic=DEFAULTS)
# run_plot(transformation_name=["min-max", "standardization", "median-quartile", "adversarial-0.5"], seed=SEED, dic=DEFAULTS)
# run_plot(p_byzantine=[0., 0.1, 0.2, 0.3, 0.4, 0.51], seed=SEED, dic=DEFAULTS)
