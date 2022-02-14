import os

from tests.compare import generate_all_data, multiple_experiments

N_PROC = os.cpu_count()

DEFAULT = {
        'n_attempts': 5, 'n_voters': 150, 'n_extreme': 0, 'n_alternatives': 300, 'density': .1,
        'noise_range': (0, 0), 'p_byzantine': 0., 'byz_density': 0, 'byz_strat': 'anti', 'voting_resilience': 5.,
        'transformation_name': "min-max", 'sm': 0, 'n_proc': N_PROC
    }

EXPERIMENTS = [
    (DEFAULT,
     {'name': 'sm', 'params': [0., 0.25, 0.5, 0.75, 1.]}
     )
]

SPARSE_EXPERIMENTS = [
        (DEFAULT,
        {'name': 'density', 'params': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]}
        )
]

if __name__ == '__main__':
    print("Total number of processes used: {}".format(N_PROC))
    ratings, mask, voting_rights, original_preferences, deltas = generate_all_data(delta=1e-10, seed=1, **DEFAULT)
    data = {'ratings': ratings, 'mask': mask, 'voting_rights': voting_rights}
    # multiple_experiments(SPARSE_EXPERIMENTS, data=data)     # to run sparsification experiments
    multiple_experiments(EXPERIMENTS, data=None)

