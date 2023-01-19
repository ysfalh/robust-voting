import os
import copy

from tests.compare import generate_all_data, multiple_experiments

N_PROC = 100
# N_PROC = os.cpu_count()

DEFAULT = {
        'n_attempts': 20, 'n_voters': 151, 'n_extreme': 0, 'extreme': 0, 'n_alternatives': 300, 'density': 0.1,
        'noise_range': (0, 0), 'p_byzantine': 0., 'byz_density': 1, 'byz_strat': 'random', 'voting_resilience': 10,
        'transformation_name': "min-max", 'sm': 0, 'distribution': 'normal', 'n_proc': N_PROC
    }
DEFAULT_EXTR = copy.deepcopy(DEFAULT)
DEFAULT_EXTR['n_extreme'] = 150  # the Byzantine voter is not counted
DEFAULT_EXTR['extreme'] = 0.8

DENS_EXP = {'name': 'density', 'params': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]}
BYZ_EXP = {'name': 'p_byzantine', 'params': [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]}
EXTR_EXP = {'name': 'extreme', 'params': [0.6, 0.7, 0.8, 0.9, 1]}


EXPERIMENTS = [
    # density
    (DEFAULT, DENS_EXP),
    (DEFAULT_EXTR, DENS_EXP),
    # p_byzantine  
    (DEFAULT, BYZ_EXP),
    (DEFAULT_EXTR, BYZ_EXP),
    # extreme
    (DEFAULT_EXTR, EXTR_EXP), 
]

# SPARSE_EXPERIMENTS = [
#         (DEFAULT,
#         {'name': 'density', 'params': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]}
#         )
# ]

RESPARSE = False

if __name__ == '__main__':
    print("Total number of processes used: {}".format(N_PROC))
    if RESPARSE:
        ratings, mask, voting_rights, original_preferences, deltas = generate_all_data(delta=1e-10, seed=1, **DEFAULT)
        data = {'ratings': ratings, 'mask': mask, 'voting_rights': voting_rights}
        multiple_experiments(SPARSE_EXPERIMENTS, data=data)     # to run sparsification experiments
    else:
        multiple_experiments(EXPERIMENTS, data=None)

