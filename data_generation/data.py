import numpy as np
import itertools
import pandas as pd

from data_generation.voting_rights import generate_voting_rights


def regularize_mask(mask, rng=None):
    n_voters, n_alternatives = mask.shape
    for j in range(n_alternatives):
        if np.all(mask[:, j] == 0):
            a = rng.integers(0, n_voters)
            mask[a, j] = 1

    for i in range(n_voters):
        if np.all(mask[i, :] == 0):
            a = rng.integers(0, n_alternatives)
            mask[i, a] = 1

    return mask


def regularize_mask_comparability(mask, ratings, sm=1., rng=None):
    n_voters, n_alternatives = mask.shape
    voter_list = list(range(n_voters-1))
    rng.shuffle(voter_list)
    for i_voter in range(len(voter_list)-1):
        voter = voter_list[i_voter]
        if np.all(ratings[voter] - ratings[voter].mean() == 0):   # list is constant, we can't do anything
            continue
        voter_alters = [i for i, x in enumerate(mask[voter, :]) if x == 1]
        for i_voterbis in range(i_voter+1, len(voter_list)):
            voterbis = voter_list[i_voterbis]
            if np.all(ratings[voterbis] - ratings[voterbis].mean() == 0):  # list is constant, we can't do anything
                continue
            voterbis_alters = [i for i, x in enumerate(mask[voterbis, :]) if x == 1]
            alter_inter = [i for i in voterbis_alters if i in voter_alters]
            comparable = False
            if len(alter_inter) >= 2:
                pairs = list(itertools.combinations(alter_inter, 2))
                i = 0
                while not comparable:
                    a, b = pairs[i]
                    comparable = (ratings[voter, b] != ratings[voter, a]) and (ratings[voterbis, b] != ratings[voterbis, a])
                    i += 1

            if not comparable:
                flip_coin = rng.binomial(1, sm, 1)
                if flip_coin != 1:
                    break
                shuff_voter_alters = voter_alters
                rng.shuffle(shuff_voter_alters)
                shuff_alter = list(range(n_alternatives))
                rng.shuffle(shuff_alter)

                a = shuff_voter_alters[0]
                mask[voterbis, a] = 1
                for b in shuff_alter:
                    if (ratings[voter, b] != ratings[voter, a]) and (ratings[voterbis, b] != ratings[voterbis, a]):
                        mask[voter, b] = mask[voterbis, b] = 1
                        break

    return mask


def generate_mask(
        n_unif, n_good, n_bad, n_alternatives,
        n_byz=1, extreme=0.8, density=0.2, byz_density=1, rng=None
):
    """ generates a mask with 3 notation styles for honest voters:
        - uniform votes
        - voting for good videos only
        - voting for bad videos only
    """
    n_voters = n_unif + n_good + n_bad + n_byz
    top = int(extreme * n_alternatives)  # number of extreme videos
    mask = np.zeros((n_voters, n_alternatives), dtype=int)  # initialize mask
    # uniform
    mask[:n_unif, :] = rng.binomial(1, density, (n_unif, n_alternatives))
    # good
    mask[n_unif: n_unif + n_good, n_alternatives - top:] = rng.binomial(1, density, (n_good, top))
    # bad
    mask[n_unif + n_good: n_unif + n_good + n_bad, :top] = rng.binomial(1, density, (n_bad, top))
    # byzantine
    mask[n_voters - n_byz:, :] = rng.binomial(1, byz_density, (n_byz, n_alternatives))
    # print(mask)
    return regularize_mask(mask, rng=rng)



def create_ortho(vect, rng):
    """ returns a vector orthogonal to vect 

    vect : np vector of last coordinate different from zero
    """
    length = len(vect)
    new_vect = np.zeros(length)
    new_vect[:-1] = rng.standard_cauchy(length - 1)
    new_vect[-1] = - new_vect[:-1] @ vect[:-1] / vect[-1]
    return new_vect


def generate_data(
        n_voters, n_extreme, n_alternatives, density,
        noise_range=(0, 0), byz_density=1, byz_strat='anti', rng=None, distribution="normal", **kwargs
):
    """ generates random original preferences, ratings by voters and a mask """
    mask = generate_mask(
        n_voters - n_extreme - 1, n_extreme//2, n_extreme//2, n_alternatives, 1,
        density=density, byz_density=byz_density, rng=rng, **kwargs
    )
    if distribution == "normal":
        original_preferences = rng.normal(size=n_alternatives)
    elif distribution == "cauchy":
        original_preferences = rng.standard_cauchy(n_alternatives)
    elif distribution == "uniform":
        original_preferences = rng.uniform(-1, 1, size=n_alternatives)
    original_preferences = (original_preferences - original_preferences.min()) / (
                            original_preferences.max() - original_preferences.min())
    original_preferences = np.sort(original_preferences)
    ratings = np.zeros((n_voters, n_alternatives))

    byzantine = n_voters - 1
    noises = rng.uniform(noise_range[0], noise_range[1], n_voters)  # different noise std for each voter
    noises[byzantine] = 1e-10  # byzantine is supposed to have no uncertainty
    for voter in range(n_voters):
        if voter == byzantine:
            if byz_strat == 'random':
                ratings[voter] = rng.standard_cauchy(n_alternatives)
            elif byz_strat == 'anti':
                ratings[voter] = - original_preferences
            elif byz_strat == 'ortho':
                ratings[voter] = create_ortho(original_preferences, rng=rng)
            elif byz_strat == 'focus':
                count = np.sum(mask[:-1], axis=0)
                for i, (true_pref, nb) in enumerate(zip(original_preferences, count)):
                    if nb < density * (n_voters - 1):
                        ratings[voter][i] = - np.sign(true_pref)
                    else:
                        ratings[voter][i] = 0
            elif byz_strat == 'focus_anti':
                count = np.sum(mask[:-1], axis=0)
                for i, (true_pref, nb) in enumerate(zip(original_preferences, count)):
                    if nb < density * (n_voters - 1):
                        ratings[voter][i] = - true_pref
                    else:
                        ratings[voter][i] = 0

        else:
            scaling = rng.lognormal(1., 1.)
            translation = rng.normal(0., 10.)
            scaling = 1e-1 if scaling == 0 else scaling
            ratings[voter] = original_preferences + rng.normal(0, noises[voter], (n_alternatives,))  # Adding noise
            ratings[voter] = scaling * ratings[voter] + translation

    return ratings, original_preferences, mask, noises / np.sqrt(2)


def sparsify_mask(input_mask, density, n_extreme=0, extreme_perc=0.3, rng=None):
    pass # DEPRECATED
#     """ sparsify input mask """
#     n_voters, n_alternatives = input_mask.shape
#     # sparsification
#     sparsity = rng.binomial(1, density, (input_mask.shape))
#     new_mask = sparsity & input_mask
#     # zeroing out all nonextreme elternatives of extreme voters
#     extremity_nb = int(extreme_perc * n_alternatives)
#     extreme_idxs = np.random.randint(0, n_voters - 1, 2 * n_extreme)
#     high_idxs, low_idxs = extreme_idxs[:n_extreme], extreme_idxs[n_extreme:]
#     new_mask[high_idxs, :extremity_nb] = 0
#     new_mask[low_idxs, extremity_nb:] = 0

#     return new_mask


def generate_all_data(
        n_voters, n_extreme, n_alternatives, noise_range,
        density, byz_density, byz_strat, sm, delta, p_byzantine,
        seed, **kwargs):
    """ generates random original preferences, ratings, mask, voting_rights """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    ratings, original_preferences, mask, deltas = generate_data(
        n_voters, n_extreme, n_alternatives, noise_range=noise_range,
        density=density, byz_density=byz_density, byz_strat=byz_strat, rng=rng, **kwargs
    )
    if delta is not None:  # if delta not custom for each user
        deltas = [delta] * n_voters
    voting_rights = generate_voting_rights(n_voters, p_byzantine, rng=rng, **kwargs)
    # voting_rights, mask = regularize_voting_rights(
    #     original_preferences, voting_rights, mask,
    #     voting_resilience=voting_resilience, sm3=sm3, sm4=sm4,
    #     n_extreme=n_extreme, rng=rng, **kwargs
    # )
    mask = regularize_mask_comparability(mask, ratings, sm=sm, rng=rng)
    return ratings, mask, voting_rights, original_preferences, deltas


def read_movielens(path='data/u.data'):
    data = pd.read_csv(path, sep="\t")
    ratings = np.zeros((len(data.uid.unique()), len(data.vid.unique())))
    for uid, vid, rating in zip(data.uid, data.vid, data.rating):
        ratings[uid - 1][vid - 1] = rating
    mask = (ratings != 0) * 1
    voting_rights = np.ones(len(mask))
    return ratings, mask, voting_rights


# def add_fake_byzantin(ratings, mask, voting_rights):
#     ratings = np.append(ratings, np.zeros((1, len(ratings[0]))), axis=0)
#     mask = np.append(mask, np.ones((1, len(ratings[0]))), axis=0)
#     voting_rights = np.append(voting_rights, [0])
