from tests.compare import multiple_experiments
import os

N_PROC = os.cpu_count()

EXPERIMENTS = [
    # ({
    #     'n_attempts': 2, 'n_voters': 61, 'n_extreme': 0, 'n_alternatives': 100, 'density': 0.1,
    #     'noise_range': (0, 1), 'p_byzantine': 0, 'byz_density': 0, 'byz_strat':'random', 'voting_resilience': 1.,
    #     'transformation_name': "median-quartile", 'pair_perc': 1, 'sm3':1, 'sm4':1
    # }, 
    #     {'name':'noise_range', 'params':[(1e-5, 1e-5), (0.05, 0.05), (0.1, 0.1)]}
    # ),
    ({
        'n_attempts': 2, 'n_voters': 61, 'n_extreme': 0, 'n_alternatives': 100, 'density': 0.2,
        'noise_range': (0, 1), 'p_byzantine': 0, 'byz_density': 0, 'byz_strat':'random', 'voting_resilience': 1.,
        'transformation_name': "median-quartile", 'pair_perc': 1, 'sm3':1, 'sm4':1
    }, 
        {'name':'noise_range', 'params':[(0.01, 0.05), (1, 2)]}
    ),
    ({
        'n_attempts': 2, 'n_voters': 61, 'n_extreme': 0, 'n_alternatives': 100, 'density': 0.2,
        'noise_range': (0, 1), 'p_byzantine': 0, 'byz_density': 0, 'byz_strat':'random', 'voting_resilience': 1.,
        'transformation_name': "median-quartile", 'pair_perc': 1, 'sm3':1, 'sm4':1, 'delta':1e-10,
    }, 
        {'name':'noise_range', 'params':[(0.01, 0.05), (1, 2)]}
    ),
]

if __name__ == '__main__':
    multiple_experiments(EXPERIMENTS)
