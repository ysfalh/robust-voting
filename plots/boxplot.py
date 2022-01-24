from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from data_generation.data import generate_data
from data_generation.voting_rights import generate_voting_rights, regularize_voting_rights
from votes.mehestan import Mehestan
from votes.basic_vote import BasicVote

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
                ratings, voting_rights, mask, 
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


def disp_boxplot(bv_corr, bv_p, mh_corr, mh_p, whis=None):
    """ display boxplot of correlations and p_values """
    figure, axis = plt.subplots(1, 2)
    # axis[0].set_ylim([0, 1])
    axis[1].set_ylim([0, 0.1])
    axis[0].set_title("correlation")
    axis[0].boxplot((bv_corr, mh_corr), whis=whis)
    axis[1].set_title("p_values")
    axis[1].boxplot((bv_p, mh_p), whis=whis)
    plt.show()
