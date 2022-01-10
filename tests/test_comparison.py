from votes.basic_vote import BasicVote
from votes.mehestan import Mehestan
from scipy.stats import pearsonr


def test_unanimity_comparison(n_attempts=20, n_voters=8, n_alternatives=16, density=.01, p_byzantine=.45,
                              byz_density=1., voting_resilience=1., transformation_name="min-max",
                              random_mask=False, seed=1234):
    bv = BasicVote(n_voters=n_voters, n_alternatives=n_alternatives, density=density, p_byzantine=p_byzantine,
                   byz_density=byz_density, voting_resilience=voting_resilience,
                   transformation_name=transformation_name, random_mask=random_mask, seed=seed)
    mh = Mehestan(n_voters=n_voters, n_alternatives=n_alternatives, density=density, p_byzantine=p_byzantine,
                  byz_density=byz_density, voting_resilience=voting_resilience,
                  transformation_name=transformation_name, random_mask=random_mask, seed=seed)

    bv_correl = 1, 0
    mh_correl = 1, 0
    j_bv, j_mh = 0, 0
    for i in range(n_attempts):
        print("{} Mehestan ATTEMPT {} {}".format("-" * 50, i, "-" * 50))
        out, original_preferences = mh.run()
        print("voting rights: {}".format(mh.voting_rights))
        # print("vote output: {}".format(out))
        # print("original preferences: {}".format(mh.original_preferences))
        if pearsonr(out, original_preferences)[0] < mh_correl[0]:
            mh_correl = pearsonr(out, original_preferences)
            j_mh = i
        print("--> pearson correlation: {}".format(pearsonr(out, original_preferences)))

        print("{} BasicVote ATTEMPT {} {}".format("-" * 50, i, "-" * 50))
        out, original_preferences = bv.run(ratings=mh.bv_ratings, original_preferences=mh.original_preferences,
                                           voting_rights=mh.voting_rights, mask=mh.mask, generate_data=False)
        print("voting rights: {}".format(bv.voting_rights))
        # print("vote output: {}".format(out))
        if pearsonr(out, original_preferences)[0] < bv_correl[0]:
            bv_correl = pearsonr(out, original_preferences)
            j_bv = i
        print("--> pearson correlation: {}".format(pearsonr(out, original_preferences)))

    print("_" * 100)
    print("Best Mehestan adversarial attempt {} with correlation {}".format(j_mh, mh_correl))
    print("Best BasicVote adversarial attempt {} with correlation {}".format(j_bv, bv_correl))


test_unanimity_comparison(n_attempts=25)
