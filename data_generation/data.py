import random

import numpy as np


def regularize_mask(mask, pair=(0, 1), pair_perc=1., rng=None):
    """ add a pair of commonly voted alternatives """
    n_voters, n_alternatives = mask.shape
    mask[:, pair[0]] = mask[:, pair[1]] = rng.binomial(1, pair_perc, (n_voters,))
    for j in range(n_alternatives):
        if np.all(mask[:, j] == 0):
            a = rng.integers(0, n_voters)
            mask[a, j] = 1

    for i in range(n_voters):
        if np.all(mask[i, :] == 0):
            a = rng.integers(0, n_alternatives)
            mask[i, a] = 1

    return mask


def generate_mask(
        n_unif, n_good, n_bad, n_alternatives,
        n_byz=1, extreme=0.3, density=0.2, byz_density=1, pair_perc=1., rng=None
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

    alts = list(range(n_alternatives))
    rng.shuffle(alts)
    meh_pair = alts[:2]
    # print("Pair chosen at generation: {}".format(meh_pair))

    return regularize_mask(mask, pair=meh_pair, pair_perc=pair_perc, rng=rng)

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
        noise_range=(0, 0), byz_density=1, byz_strat='anti', pair_perc=1., rng=None
    ):
    """ generates random original preferences, ratings by voters and a mask """

    original_preferences = rng.standard_cauchy(n_alternatives)
    original_preferences = (original_preferences - original_preferences.min()) / (
            original_preferences.max() - original_preferences.min())
    original_preferences = np.sort(original_preferences)
    ratings = np.zeros((n_voters, n_alternatives))

    byzantine = n_voters - 1  # TODO parametrize nb byzantine
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

        else:
            scaling = rng.lognormal(1., 1.)
            translation = rng.normal(0., 10.)
            scaling = 1e-1 if scaling == 0 else scaling
            ratings[voter] = original_preferences + rng.normal(0, noises[voter], (n_alternatives,))  # Adding noise
            ratings[voter] = scaling * ratings[voter] + translation

    mask = generate_mask(
        n_voters - 2 * n_extreme - 1, n_extreme, n_extreme, n_alternatives, 1,
        density=density, byz_density=byz_density, pair_perc=pair_perc, rng=rng
    )

    return ratings, original_preferences, mask, noises / np.sqrt(2)
