from lib2to3.pgen2.pgen import DFAState
from tests.compare import multiple_experiments
import os

# TODO: make each attempt with a different seed chosen from a predefined list of seeds
# TODO: make n_subattempts for each generated original_preferences
N_PROC = os.cpu_count()
DEFAULTS = {
    'n_attempts': 2, 'n_voters': 61, 'n_extreme': 0, 'n_alternatives': 100, 'density': 0.1, 'delta': 1e-6,
    'noise': 0, 'p_byzantine': 0, 'byz_density': 0, 'byz_strat':'random', 'voting_resilience': 1.,
    'transformation_name': "median-quartile", 'pair_perc': 1, 'sm3':1, 'sm4':1
}


EXPERIMENTS = {
    # 'p_byzantine':[0., 0.51],
    'delta':[1e-13, 0.01, 0.1, 1],
    # 'pair_perc':[0.01, 0.05, 0.1]

}

EXPERIMENTS = [
    ({
        'n_attempts': 2, 'n_voters': 61, 'n_extreme': 0, 'n_alternatives': 100, 'density': 0.1, 'delta': 1e-6,
        'noise': 0, 'p_byzantine': 0, 'byz_density': 0, 'byz_strat':'random', 'voting_resilience': 1.,
        'transformation_name': "median-quartile", 'pair_perc': 1, 'sm3':1, 'sm4':1
    }, 
        {'name':'noise', 'params':[0,0.05,0.1]}
    ),
    ({
        'n_attempts': 2, 'n_voters': 61, 'n_extreme': 0, 'n_alternatives': 30, 'density': 1, 'delta': 1e-6,
        'noise': 0, 'p_byzantine': 0, 'byz_density': 0, 'byz_strat':'random', 'voting_resilience': 1.,
        'transformation_name': "median-quartile", 'pair_perc': 1, 'sm3':1, 'sm4':1
    }, 
        {'name':'noise', 'params':[0.5, 1, 2]}
    ),

]

if __name__ == '__main__':
    multiple_experiments(EXPERIMENTS)
