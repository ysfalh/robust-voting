from votes.basic_vote import BasicVote
from votes.mehestan import Mehestan
from scipy.stats import pearsonr


def test_unanimity(vote_name, n_attempts=20, n_voters=5, n_alternatives=5, density=.1, p_byzantine=.3,
                   byz_density=.5, voting_resilience=.01, random_mask=False):
    if vote_name == "BasicVote":
        bv = BasicVote(n_voters=n_voters, n_alternatives=n_alternatives, density=density, p_byzantine=p_byzantine,
                       byz_density=byz_density, voting_resilience=voting_resilience, random_mask=random_mask)
    elif vote_name == "Mehestan":
        bv = Mehestan(n_voters=n_voters, n_alternatives=n_alternatives, density=density, p_byzantine=p_byzantine,
                      byz_density=byz_density, voting_resilience=voting_resilience, random_mask=random_mask)

    correl = 1, 0
    j = 0
    for i in range(n_attempts):
        print("{} ATTEMPT {} {}".format("-" * 50, i, "-" * 50))
        out, original_preferences = bv.run()
        print("voting rights: {}".format(bv.voting_rights))
        print("vote output: {}".format(out))
        if pearsonr(out, original_preferences)[0] < correl[0]:
            correl = pearsonr(out, original_preferences)
            j = i
        print("--> pearson correlation: {}".format(pearsonr(out, original_preferences)))

    print("_" * 100)
    print("Best adversarial attempt {} with correlation {}".format(j, correl))


# name = "BasicVote"
name = "Mehestan"
test_unanimity(name, n_attempts=5)
