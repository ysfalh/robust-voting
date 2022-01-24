from numpy.random import default_rng

from plots.boxplot import disp_boxplot, comparative_runs


SEED = 4
# OLD = True
RNG = default_rng(SEED)

bv_corr, bv_p, mh_corr, mh_p = comparative_runs(
    n_attempts=1, n_voters=5, n_extreme=0, n_alternatives=10, 
    density=0.2, noise=0.1, byz_density=1, regularize=True, p_byzantine=0.4, voting_resilience=1,
    rng=RNG
)
disp_boxplot(bv_corr, bv_p, mh_corr, mh_p)