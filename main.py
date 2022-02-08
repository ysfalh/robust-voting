import os

from tests.compare import generate_all_data, multiple_experiments
from data_generation.data import read_movielens, add_fake_byzantin

N_PROC = os.cpu_count()

DEFAULT = {
        'n_attempts': 2, 'n_voters': 21, 'n_extreme': 0, 'n_alternatives': 20, 'density': 1,
        'noise_range': (0, 0), 'p_byzantine': 0, 'byz_density': 1, 'byz_strat':'random', 'voting_resilience': 1.,
        'transformation_name': "median-quartile", 'pair_perc': 1, 'sm3':0, 'sm4':0
    }

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

SPARSE_EXPERIMENTS = [
        ({
            'density': 0.3, 'n_extreme':0,
            'n_attempts':1,  'delta':1e-10, 'voting_resilience':1., 'transformation_name':"min-max", 'n_proc':1
    }, 
        {'name':'voting_resilience', 'params':[1, 10]}
        )
]

if __name__ == '__main__':

    # multiple_experiments(EXPERIMENTS)
    ratings, mask, voting_rights, original_preferences, deltas = generate_all_data(delta=1e-10, seed=1, **DEFAULT)
    # ratings, mask, voting_rights = read_movielens(path='data/u.data')
    data = {'ratings': ratings, 'mask': mask, 'voting_rights': voting_rights}
    multiple_experiments(SPARSE_EXPERIMENTS, data=data)

