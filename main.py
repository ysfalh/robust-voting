from numpy.random import default_rng
from tests.compare import run_plot


SEED = 4
RNG = default_rng(SEED)

DEFAULTS = {'n_attempts':5, 'n_voters':10, 'n_extreme':0, 'n_alternatives':30, 'density':0.1,    
    'noise':0, 'p_byzantine':0.4, 'byz_density':1., 'voting_resilience':1.,
     'transformation_name':"min-max", 'regularize':True,}

run_plot(density=[0.05, 0.1, 0.2, 0.3], rng=RNG, dic=DEFAULTS)
