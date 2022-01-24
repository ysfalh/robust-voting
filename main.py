from numpy.random import default_rng
from plots.boxplot import disp_boxplot
from tests.compare import comparative_runs


SEED = 4
RNG = default_rng(SEED)

bv_corr, bv_p, mh_corr, mh_p = comparative_runs(
    n_attempts=10, n_voters=20, n_extreme=0, n_alternatives=100,
    density=0.1, noise=0., byz_density=1, regularize=True, p_byzantine=0.3, voting_resilience=1,
    rng=RNG
)
disp_boxplot(bv_corr, bv_p, mh_corr, mh_p, whis=float("inf"), labels=["BasicVote", "Mehestan"])