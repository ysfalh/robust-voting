from scipy.stats import pearsonr
from data_generation.data import generate_data
from data_generation.voting_rights import generate_voting_rights, regularize_voting_rights
from votes.mehestan import Mehestan
from votes.basic_vote import BasicVote
from plots.boxplot import draw_curves, range_boxplot


def comparative_runs(
        n_attempts, n_voters, n_extreme, n_alternatives,
        density=.01, noise=0, p_byzantine=.45, byz_density=1., voting_resilience=1.,
        transformation_name="min-max", regularize=True, **kwargs
):
    """ comparing the voting algorithms on generated data """

    bv_corr, bv_p, mh_corr, mh_p = [], [], [], []
    for i in range(n_attempts):

        # data generation
        ratings, original_preferences, mask = generate_data(
            n_voters, n_extreme, n_alternatives, noise=noise,
            density=density, byz_density=byz_density, regularize=regularize, **kwargs
        )
        voting_rights = generate_voting_rights(n_voters, p_byzantine, **kwargs)
        if regularize:
            voting_rights, mask = regularize_voting_rights(
                original_preferences, voting_rights, mask,
                voting_resilience=voting_resilience, **kwargs
            )

        # voting with Mehestan
        mh = Mehestan(ratings, mask, voting_rights, voting_resilience)
        out = mh.run()
        corr, pval = pearsonr(out, original_preferences)
        mh_corr.append(corr)
        mh_p.append(pval)

        # voting with Basic Vote
        bv = BasicVote(
            ratings, mask, voting_rights,
            voting_resilience, transformation_name=transformation_name
        )
        out = bv.run()

        corr, pval = pearsonr(out, original_preferences)
        bv_corr.append(corr)
        bv_p.append(pval)

    return bv_corr, bv_p, mh_corr, mh_p


def auto_run(seed=None, dic={}, rng=None, **kwargs):
    """ multiple runs of both algorithms with 1 parameter changing """
    l_bv_corr, l_bv_p, l_mh_corr, l_mh_p = [], [], [], []
    for name, values in kwargs.items():  # only 1 iteration
        for param in values:
            print(name, ':', param)
            dic.pop(name, None)  # remove parameter default value
            bv_corr, bv_p, mh_corr, mh_p = comparative_runs(**{name:param}, **dic, rng=rng)
            l_bv_corr.append(bv_corr)
            l_bv_p.append(bv_p)
            l_mh_corr.append(mh_corr)
            l_mh_p.append(mh_p)
    return l_bv_corr, l_bv_p, l_mh_corr, l_mh_p


def run_plot(seed=None, dic={}, rng=None, **kwargs):
    """ takes default parameters as a dictionnary and one parameter list 
        and plots results with that parameter changing
    """
    for name, values in kwargs.items():  # only 1 iteration
        l_bv_corr, l_bv_p, l_mh_corr, l_mh_p = auto_run(seed=seed, dic=dic, rng=rng, **kwargs)
        draw_curves(l_bv_corr, l_mh_corr, values, labels=('Basic Vote', 'Mehestan'), x_name=name)
        range_boxplot(l_bv_corr, values, title='Basic Vote', x_name=name)
        range_boxplot(l_mh_corr, values, title='Mehestan', x_name=name)